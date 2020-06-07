use image::{DynamicImage, GenericImage, GrayAlphaImage, GrayImage, Pixel};
use imageproc;
use imageproc::geometric_transformations::*;
use rand::prelude::*;

#[derive(Clone)]
pub struct FloatImage {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl FloatImage {
    pub fn to_image(&self) -> DynamicImage {
        let mut img = DynamicImage::new_luma8(self.width, self.height);
        for x in 0..self.width {
            for y in 0..self.height {
                let v = self.values[(x * self.height + y) as usize];
                let v = (v.min(1.0).powf(1.0 / 2.2) * 255.0) as u8;
                img.put_pixel(x, y, Pixel::from_channels(v, v, v, 255));
            }
        }
        img
    }
}

// Simple CDF
// Can be loaded from image or can be automatic
// When automatic, the image get blured.
// starting after some percent (20%) in the orginal code
// then use some ramping to get a "t" value.
// Then use a guassian based on some percent
pub struct CDF {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl CDF {
    pub fn from_image(filename: &str) -> CDF {
        let img = image::open(filename).unwrap().to_luma();

        // Create the new CDF
        let mut cdf = Vec::with_capacity((img.width() * img.height()) as usize + 1);
        let mut cur = 0.0;
        for p in img.pixels() {
            cdf.push(cur);
            cur += p[0] as f32 / 255.0;
        }
        cdf.push(cur);

        // Normalize the cdf
        cdf.iter_mut().for_each(|x| *x /= cur);

        CDF {
            values: cdf,
            width: img.width(),
            height: img.height(),
        }
    }

    pub fn sample(&self, v: f32) -> (u32, u32) {
        assert!(v >= 0.0);
        assert!(v < 1.0);

        let id = match self
            .values
            .binary_search_by(|probe| probe.partial_cmp(&v).unwrap())
        {
            Ok(x) => x,
            Err(x) => x - 1,
        };

        let x = id as u32 / self.height;
        let y = id as u32 % self.height;
        (x, y)
    }
}

pub struct GradientImage {
    mag: FloatImage,
    angle: FloatImage,
}

impl GradientImage {
    pub fn new(img: &image::GrayImage) -> Self {
        let w = img.width() as u32;
        let h = img.height() as u32;

        let gx = imageproc::gradients::horizontal_sobel(&img);
        let gy = imageproc::gradients::vertical_sobel(&img);

        let mut mag = vec![0.0; (w * h) as usize];
        let mut angle = vec![0.0; (w * h) as usize];
        for x in 0..img.width() {
            for y in 0..img.height() {
                let xv = gx.get_pixel(x, y)[0] as f32;
                let yv = gy.get_pixel(x, y)[0] as f32;

                mag[(x * h + y) as usize] = (xv * xv + yv * yv).sqrt();
                angle[(x * h + y) as usize] = yv.atan2(xv).to_degrees();
            }
        }
        let mag_max = *mag.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap();
        let mag = mag
            .into_iter()
            .map(|v| (v / mag_max).powf(0.3))
            .collect::<Vec<_>>();

        GradientImage {
            mag: FloatImage {
                values: mag,
                width: w,
                height: h,
            },
            angle: FloatImage {
                values: angle,
                width: w,
                height: h,
            },
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.mag.width, self.mag.height)
    }

    pub fn get(&self, x: u32, y: u32) -> (f32, f32) {
        let id = (x * self.mag.height + y) as usize;
        (self.mag.values[id], self.angle.values[id])
    }

    pub fn get_safe(&self, x: i32, y: i32) -> (f32, f32) {
        if x >= 0 && x < self.mag.width as i32 && y >= 0 && y < self.mag.height as i32 {
            self.get(x as u32, y as u32)
        } else {
            (0.0, 0.0)
        }
    }

    pub fn to_image(&self) -> (DynamicImage, DynamicImage) {
        (self.mag.to_image(), self.angle.to_image())
    }
}

pub struct MinMax {
    pub min: f32,
    pub max: f32,
}
impl MinMax {
    pub fn value(&self, t: f32) -> f32 {
        assert!(t >= 0.0 && t <= 1.0);
        (self.max - self.min) * t + self.min
    }
    pub fn mix(begin: &MinMax, end: &MinMax, time: f32) -> Self {
        assert!(time >= 0.0 && time <= 1.0);
        Self {
            min: end.min * time + begin.min * (1.0 - time),
            max: end.max * time + begin.max * (1.0 - time),
        }
    }
}

pub struct GeneticDrawing {
    pub img_grey: GrayImage,
    pub img_gradient: GradientImage,
    pub bruch_range: (MinMax, MinMax),
    pub brushes: Vec<GrayAlphaImage>,
}

impl GeneticDrawing {
    pub fn load(filename: &str) -> Self {
        // Load image
        let img_grey = image::open(filename).unwrap().into_luma();
        let img_gradient = GradientImage::new(&img_grey);
        Self {
            img_grey,
            img_gradient,
            bruch_range: (MinMax { min: 0.3, max: 0.7 }, MinMax { min: 0.1, max: 0.3 }),
            brushes: vec![],
        }
    }

    pub fn register_brush(&mut self, filename: &str) {
        let mut img_grey = image::open(filename).unwrap().into_luma_alpha();
        img_grey.pixels_mut().for_each(|p| {
            if p[0] == 0 {
                p[1] = 0;
            }
        });
        // TODO: Make it squared.
        self.brushes.push(img_grey);
    }

    pub fn clear_brushes(&mut self) {
        self.brushes.clear();
    }
}

pub struct DNAContext<'draw, 'cdf> {
    org_image: GrayImage,
    pub image: GrayImage,
    offset: i32,
    strokes: Vec<Stroke>,
    strokes_scales: MinMax,
    error: f32,
    pub gen: &'draw GeneticDrawing,
    pub pos_cdf: Option<&'cdf CDF>,
}

pub fn blend_alpha(dest: &mut GrayImage, org: &GrayAlphaImage, xoff: i32, yoff: i32) {
    // Safe version of image blending
    for x in 0..org.width() {
        for y in 0..org.height() {
            let x_dest = x as i32 + xoff;
            let y_dest = y as i32 + yoff;
            if x_dest >= 0
                && x_dest < dest.width() as i32
                && y_dest >= 0
                && y_dest < dest.height() as i32
            {
                let porg = org.get_pixel(x, y);
                let alpha = porg[1] as f32 / 255.0;
                let pdest = dest.get_pixel_mut(x_dest as u32, y_dest as u32);
                pdest[0] = (alpha * porg[0] as f32 + (1.0 - alpha) * pdest[0] as f32) as u8;
            }
        }
    }
}

impl<'draw, 'cdf> DNAContext<'draw, 'cdf> {
    pub fn new(
        gen: &'draw GeneticDrawing,
        brushcount: usize,
        rng: &mut rand::rngs::ThreadRng,
        time: f32,
        prev_image: Option<&GrayImage>,
        pos_cdf: Option<&'cdf CDF>,
    ) -> Self {
        // Compute the max size bruches
        let strokes_scales = MinMax::mix(&gen.bruch_range.0, &gen.bruch_range.1, time);
        let max_size_bruches = strokes_scales.value(1.0);
        let offset = gen
            .brushes
            .iter()
            .map(|b| (b.width().max(b.height()) as f32 * max_size_bruches))
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap();
        let offset = (offset * 0.5) as i32;

        // Create a bigger image for easier spatting
        let mut image =
            image::DynamicImage::new_luma8(gen.img_grey.width(), gen.img_grey.height()).into_luma();
        if let Some(prev_image) = prev_image {
            // If provided, initialize with previous iteration
            image
                .pixels_mut()
                .zip(prev_image.pixels())
                .for_each(|(p0, p1)| p0[0] = p1[0]);
        }
        let org_image = image.clone();

        // Create initial strokes and fill the image
        let strokes = (0..brushcount)
            .map(|_| {
                Stroke::new(
                    rng,
                    &strokes_scales,
                    (image.width(), image.height()),
                    offset,
                    gen.brushes.len(),
                    Some(&gen.img_gradient),
                    pos_cdf,
                )
            })
            .collect::<Vec<_>>();
        for s in &strokes {
            blend_alpha(&mut image, &s.draw(&gen.brushes), s.pos.0, s.pos.1);
        }

        let mut dna = DNAContext {
            org_image,
            image,
            offset,
            strokes,
            strokes_scales,
            error: 0.0,
            gen,
            pos_cdf,
        };
        dna.error = dna.compute_error(&dna.image);
        dna
    }

    pub fn iterate(&mut self, number_iter: usize, rng: &mut rand::rngs::ThreadRng) {
        for _it in 0..number_iter {
            for i in 0..self.strokes.len() {
                // Clone the original image
                let mut new_image = self.org_image.clone();
                // Mutate one stroke
                let mut new_strokes = self.strokes.clone();
                new_strokes[i] = self.strokes[i].mutate(
                    rng,
                    &self.strokes_scales,
                    (self.image.width(), self.image.height()),
                    self.offset,
                    self.gen.brushes.len(),
                    Some(&self.gen.img_gradient),
                    self.pos_cdf,
                );
                // Update the images with all strokes
                for s in &new_strokes {
                    blend_alpha(&mut new_image, &s.draw(&self.gen.brushes), s.pos.0, s.pos.1);
                }
                // Compute the error
                let new_error = self.compute_error(&new_image);
                if self.error > new_error {
                    self.error = new_error;
                    self.strokes = new_strokes;
                    self.image = new_image;
                }
            }
            // DEBUG
            // dbg!(self.error);
            // self.image.save(&format!("{}.png", _it)).unwrap();
        }
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn to_image(self) -> GrayImage {
        self.image
    }

    fn compute_error(&self, image: &GrayImage) -> f32 {
        image
            .pixels()
            .zip(self.gen.img_grey.pixels())
            .map(|(p0, p1)| {
                let p_diff = p0[0] as f32 - p1[0] as f32;
                p_diff.abs() / 255.0
            })
            .sum()
    }
}

// This object represent painting strokes
#[derive(Clone, Debug)]
pub struct Stroke {
    pub value: f32,
    pub size: f32,
    pub pos: (i32, i32),
    pub rotation: f32,
    pub brush_id: usize,
}

impl Stroke {
    fn gen_rotation(&self, rng: &mut rand::rngs::ThreadRng, grads: Option<&GradientImage>) -> f32 {
        match grads {
            None => rng.gen_range(0.0, 360.0),
            Some(ref v) => {
                let (mag, angle) = v.get_safe(self.pos.0, self.pos.1);
                let angle = angle + 90.0;
                rng.gen_range(-180.0, 180.0) * (1.0 - mag) + angle
            }
        }
    }

    fn gen_position(
        rng: &mut rand::rngs::ThreadRng,
        img_size: (u32, u32),
        offset: i32,
        cdf: Option<&CDF>,
    ) -> (i32, i32) {
        match cdf {
            None => (
                rng.gen_range(-offset, img_size.0 as i32 + offset),
                rng.gen_range(-offset, img_size.1 as i32 + offset),
            ),
            Some(ref cdf) => {
                let (x, y) = cdf.sample(rng.gen_range(0.0, 1.0));
                (x as i32, y as i32)
            }
        }
    }

    pub fn new(
        rng: &mut rand::rngs::ThreadRng,
        scale: &MinMax,
        img_size: (u32, u32),
        offset: i32,
        nb_bruches: usize,
        grads: Option<&GradientImage>,
        cdf: Option<&CDF>,
    ) -> Stroke {
        let value = rng.gen_range(0.0, 1.0);
        let size = scale.value(rng.gen_range(0.0, 1.0));
        let pos = Stroke::gen_position(rng, img_size, offset, cdf);
        let rotation = rng.gen_range(0.0, 360.0);
        let brush_id = rng.gen_range(0, nb_bruches);
        // Update the rotation
        let mut s = Stroke {
            value,
            size,
            pos,
            rotation,
            brush_id,
        };
        s.rotation = s.gen_rotation(rng, grads);
        s
    }

    pub fn mutate(
        &self,
        rng: &mut rand::rngs::ThreadRng,
        scale: &MinMax,
        img_size: (u32, u32),
        offset: i32,
        nb_bruches: usize,
        grads: Option<&GradientImage>,
        cdf: Option<&CDF>,
    ) -> Stroke {
        let mut mutations = vec![0, 1, 2, 3, 4];
        mutations.shuffle(rng);
        let nb_mutations = rng.gen_range(1, mutations.len() + 1);
        mutations.resize(nb_mutations, -1);
        mutations.sort();

        let mut new_stroke = self.clone();
        for m in mutations {
            match m {
                0 => new_stroke.value = rng.gen_range(0, 2) as f32,
                1 => new_stroke.size = scale.value(rng.gen_range(0.0, 1.0)),
                2 => new_stroke.pos = Stroke::gen_position(rng, img_size, offset, cdf),
                3 => new_stroke.rotation = self.gen_rotation(rng, grads),
                4 => new_stroke.brush_id = rng.gen_range(0, nb_bruches),
                _ => panic!("Not covered mutation approach"),
            };
        }

        new_stroke
    }

    pub fn draw(&self, brushes: &Vec<GrayAlphaImage>) -> GrayAlphaImage {
        let b = &brushes[self.brush_id];
        let trans = Projection::translate(b.width() as f32 * 0.5, b.height() as f32 * 0.5)
            * Projection::rotate(self.rotation.to_radians())
            * Projection::scale(self.size, self.size);
        let b = imageproc::geometric_transformations::warp(
            b,
            &trans,
            imageproc::geometric_transformations::Interpolation::Bilinear,
            image::LumaA([0, 0]),
        );
        // Crop the image using bounding box
        let pos = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let pos = pos
            .into_iter()
            .map(|(x, y)| trans * (x * b.width() as f32, y * b.height() as f32))
            .collect::<Vec<_>>();
        let xmin = pos
            .iter()
            .map(|p| p.0)
            .min_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .max(0.0) as u32;
        let ymin = pos
            .iter()
            .map(|p| p.1)
            .min_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .max(0.0) as u32;
        let xmax = pos
            .iter()
            .map(|p| p.0)
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .min(b.height() as f32) as u32;
        let ymax = pos
            .iter()
            .map(|p| p.1)
            .max_by(|i, j| i.partial_cmp(j).unwrap())
            .unwrap()
            .min(b.width() as f32) as u32;
        let mut b = image::imageops::crop_imm(&b, xmin, ymin, xmax - xmin, ymax - ymin).to_image();
        // Use the color and use alpha of the bruch
        b.pixels_mut().for_each(|p| {
            p[1] = p[0];
            p[0] = (self.value * 255.0) as u8;
        });
        b
    }
}
