use image::{DynamicImage, GenericImage, GrayAlphaImage, GrayImage, Pixel};
use imageproc;
use imageproc::geometric_transformations::*;
use rand::prelude::*;

/// Helper function to transform Vec floats to GrayScale image (image crate)
fn vec_f32_to_image(values: &Vec<f32>, width: u32, height: u32, scale: f32) -> DynamicImage {
    let mut img = DynamicImage::new_luma8(width, height);
    for x in 0..width {
        for y in 0..height {
            let v = values[(y * width + x) as usize] * scale;
            let v = (v.max(0.0).min(1.0) * 255.0) as u8;
            img.put_pixel(x, y, Pixel::from_channels(v, v, v, 255));
        }
    }
    img
}

/// Sturcture to represent an image as floating point values
/// This is usefull for storing gradient magnitude and orientation
/// The internal mapping of values following the image crate convention
#[derive(Clone)]
pub struct FloatImage {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl FloatImage {
    pub fn to_image(&self) -> DynamicImage {
        vec_f32_to_image(&self.values, self.width, self.height, 1.0)
    }
}

/// Structure to store a CDF of image values (via FloatImage)
/// This structure is used to model custom distribution on image-space
pub struct Distribution {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl Distribution {
    /// Constructor that takes a grayimage and compute its associated CDF
    pub fn from_image(img: &GrayImage) -> Distribution {
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

        Distribution {
            values: cdf,
            width: img.width(),
            height: img.height(),
        }
    }

    /// Constructor that uses gradient magnitude float map to compute the CDF
    /// Note that the gradient information get filter with a gaussian kernel
    /// before building the CDF
    pub fn from_gradients(image: &DynamicImage, gaussian_size: f32) -> Distribution {
        // Compute gradient
        let img = image.to_luma();
        let gaussian_size = img.width() as f32 * gaussian_size;
        let img = imageproc::filter::gaussian_blur_f32(&img, gaussian_size);
        Self::from_image(&img)
    }

    /// Function that return a image position
    /// using v random number v in [0; 1]
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

        let y = id as u32 / self.width;
        let x = id as u32 % self.width;
        (x, y)
    }

    pub fn to_image(&self) -> DynamicImage {
        let mut values = vec![0.0; (self.width * self.height) as usize];
        for i in 0..(self.width * self.height) as usize {
            values[i] = self.values[i + 1] - self.values[i];
        }
        vec_f32_to_image(
            &values,
            self.width,
            self.height,
            0.5 * (self.width * self.height) as f32,
        )
    }
}

/// Structure that represent the image gradient
/// The gradient are store as magnitude and orientation
/// as these two information are used when generating strokes
pub struct GradientImage {
    pub mag: FloatImage,
    pub angle: FloatImage,
}

impl GradientImage {
    pub fn new(img: &image::GrayImage) -> Self {
        let w = img.width() as u32;
        let h = img.height() as u32;

        // Compute gradients
        // TODO: In the original code, it uses K=1 sobel filter.
        //  In this case we computing sobel K=3 filters
        let gx = imageproc::gradients::horizontal_sobel(&img);
        let gy = imageproc::gradients::vertical_sobel(&img);

        // Compute magnitude and angle
        let mut mag = vec![0.0; (w * h) as usize];
        let mut angle = vec![0.0; (w * h) as usize];
        for (i, (xv, yv)) in gx.pixels().zip(gy.pixels()).enumerate() {
            let xv = xv[0] as f32;
            let yv = yv[0] as f32;
            mag[i] = (xv * xv + yv * yv).sqrt();
            angle[i] = yv.atan2(xv).to_degrees();
        }

        // Normalize the magnitude
        let mag_max = *mag.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap();

        // Do a transform to keep lower gradients as well
        // TODO: Explore or expose this parameter
        //  As some users want to make more aggressive approaches
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
        let id = (y * self.mag.width + x) as usize;
        (self.mag.values[id], self.angle.values[id])
    }

    pub fn get_safe(&self, x: i32, y: i32) -> (f32, f32) {
        if x >= 0 && x < self.mag.width as i32 && y >= 0 && y < self.mag.height as i32 {
            self.get(x as u32, y as u32)
        } else {
            // If it is outside the image
            // the safest option is to return no information
            (0.0, 0.0)
        }
    }

    pub fn to_image(&self) -> (DynamicImage, DynamicImage) {
        (self.mag.to_image(), self.angle.to_image())
    }
}

/// Structure to represent brushes scales
pub struct ScaleRange {
    pub min: f32,
    pub max: f32,
}
impl ScaleRange {
    pub fn value(&self, t: f32) -> f32 {
        assert!(t >= 0.0 && t <= 1.0);
        (self.max - self.min) * t + self.min
    }

    /// Interpolation between two brush scales (time in [0; 1])
    pub fn mix(begin: &ScaleRange, end: &ScaleRange, time: f32) -> Self {
        assert!(time >= 0.0 && time <= 1.0);
        ScaleRange {
            min: end.min * time + begin.min * (1.0 - time),
            max: end.max * time + begin.max * (1.0 - time),
        }
    }
}

/// Structure representing the whole drawing process
pub struct GeneticDrawing {
    pub img_grey: GrayImage,
    pub img_gradient: GradientImage,
    pub bruch_range: (ScaleRange, ScaleRange),
    pub brushes: Vec<GrayAlphaImage>,
}

impl GeneticDrawing {
    pub fn load(img: DynamicImage) -> Self {
        // Load image
        let img_grey = img.into_luma();
        let img_gradient = GradientImage::new(&img_grey);
        Self {
            img_grey,
            img_gradient,
            // Default parameter from the original repository
            bruch_range: (
                ScaleRange { min: 0.3, max: 0.7 },
                ScaleRange { min: 0.1, max: 0.3 },
            ),
            brushes: vec![],
        }
    }

    pub fn register_brush(&mut self, img: DynamicImage) {
        let mut img_grey = img.into_luma_alpha();
        img_grey.pixels_mut().for_each(|p| {
            if p[0] == 0 {
                p[1] = 0;
            }
        });
        self.brushes.push(img_grey);
    }

    pub fn clear_brushes(&mut self) {
        self.brushes.clear();
    }
}

/// Structure representing one iteration
/// of the genetic algorithm
pub struct DNAContext<'draw, 'cdf> {
    org_image: GrayImage,
    pub image: GrayImage,
    strokes: Vec<Stroke>,
    strokes_scales: ScaleRange,
    error: f32,
    /// The original drawing.
    /// We do a borrowing to avoid the data changing during the genetic optimization
    pub gen: &'draw GeneticDrawing,
    /// The image distribution if provided.
    /// If missing, we use a uniform distribution to generate brush position.
    pub pos_cdf: Option<&'cdf Distribution>,
}

/// Custom function to blend a brush (org) inside the main image
fn blend_alpha(dest: &mut GrayImage, org: &GrayAlphaImage, (xoff, yoff): (i32, i32)) {
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
        pos_cdf: Option<&'cdf Distribution>,
    ) -> Self {
        // Compute the max size bruches
        let strokes_scales = ScaleRange::mix(&gen.bruch_range.0, &gen.bruch_range.1, time);

        // Create a image for the splatting
        let mut image =
            image::DynamicImage::new_luma8(gen.img_grey.width(), gen.img_grey.height()).into_luma();
        if let Some(prev_image) = prev_image {
            // If provided, initialize with previous iteration
            image
                .pixels_mut()
                .zip(prev_image.pixels())
                .for_each(|(p0, p1)| p0[0] = p1[0]);
        }
        // We need to keep the original version of the provided initial image
        // as the optimization procedure will write inside the image attribute
        let org_image = image.clone();

        // Create initial strokes and fill the image
        let strokes = (0..brushcount)
            .map(|_| {
                Stroke::new(
                    rng,
                    &strokes_scales,
                    (image.width(), image.height()),
                    &gen.brushes,
                    Some(&gen.img_gradient),
                    pos_cdf,
                )
            })
            .collect::<Vec<_>>();
        for s in &strokes {
            blend_alpha(&mut image, &s.draw(&gen.brushes), s.get_position());
        }

        // Construct the context
        // and compute the error
        let mut dna = DNAContext {
            org_image,
            image,
            strokes,
            strokes_scales,
            error: 0.0,
            gen,
            pos_cdf,
        };
        dna.error = dna.compute_error(&dna.image);
        dna
    }

    /// Main entry point to update the strokes
    /// Each strokes are updated iteratively.
    /// The number of iteration is a sensitive parameter as it balance quality
    /// of the optimization and compute time.
    pub fn iterate(&mut self, number_iter: usize, rng: &mut rand::rngs::ThreadRng) {
        for _it in 0..number_iter {
            for i in 0..self.strokes.len() {
                // Clone the original image
                let mut new_image = self.org_image.clone();

                // Mutate one stroke
                let new_strokes = self.strokes[i].mutate(
                    rng,
                    &self.strokes_scales,
                    (self.image.width(), self.image.height()),
                    &self.gen.brushes,
                    Some(&self.gen.img_gradient),
                    self.pos_cdf,
                );

                // Update the images with all strokes (except the new one)
                for s in 0..self.strokes.len() {
                    if s == i {
                        // In this case, we draw the new stroke
                        blend_alpha(
                            &mut new_image,
                            new_strokes.raster.as_ref().unwrap(),
                            new_strokes.get_position(),
                        );
                    } else {
                        // Otherwise we draw the normal one
                        let s = &self.strokes[s];
                        blend_alpha(&mut new_image, s.raster.as_ref().unwrap(), s.get_position());
                    }
                }

                // Compute the error
                let new_error = self.compute_error(&new_image);
                if self.error > new_error {
                    self.error = new_error;
                    self.strokes[i] = new_strokes;
                    self.image = new_image;
                }
            }
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

/// Represent a stroke
#[derive(Clone, Debug)]
pub struct Stroke {
    pub value: f32,
    pub size: f32,
    pub pos: (i32, i32),
    pub rotation: f32,
    pub brush_id: usize,
    /// We keep a image of the stroke as a cache value
    /// This avoid to recompute the transformation during the optimization procedure
    pub raster: Option<GrayAlphaImage>,
}

impl Stroke {
    /// Generate the stroke rotation
    /// Note that if a gradient is provided, we will use the gradient magnitude
    /// and orientation to place the stoke
    fn gen_rotation(&self, rng: &mut rand::rngs::ThreadRng, grads: Option<&GradientImage>) -> f32 {
        match grads {
            None => rng.gen_range(0.0, 360.0),
            Some(ref v) => {
                // We favor stroke that follows the gradients
                let (mag, angle) = v.get_safe(self.pos.0, self.pos.1);
                let angle = angle + 90.0;
                rng.gen_range(-180.0, 180.0) * (1.0 - mag) + angle
            }
        }
    }

    /// Generate a stroke position
    /// If distribution is not provided, we use uniform sampling
    /// over the image space.
    fn gen_position(
        rng: &mut rand::rngs::ThreadRng,
        img_size: (u32, u32),
        cdf: Option<&Distribution>,
    ) -> (i32, i32) {
        match cdf {
            None => (
                rng.gen_range(0, img_size.0 as i32),
                rng.gen_range(0, img_size.1 as i32),
            ),
            Some(ref cdf) => {
                let (x, y) = cdf.sample(rng.gen_range(0.0, 1.0));
                (x as i32, y as i32)
            }
        }
    }

    pub fn new(
        rng: &mut rand::rngs::ThreadRng,
        scale: &ScaleRange,
        img_size: (u32, u32),
        brushes: &Vec<GrayAlphaImage>,
        grads: Option<&GradientImage>,
        cdf: Option<&Distribution>,
    ) -> Stroke {
        let value = rng.gen_range(0.0, 1.0);
        let size = scale.value(rng.gen_range(0.0, 1.0));
        let pos = Stroke::gen_position(rng, img_size, cdf);
        let rotation = rng.gen_range(0.0, 360.0);
        let brush_id = rng.gen_range(0, brushes.len());
        // Update the rotation
        let mut s = Stroke {
            value,
            size,
            pos,
            rotation,
            brush_id,
            raster: None,
        };
        // This is a bit ugly but we need to know the
        // stroke position to generate a rotation (if gradient are provided)
        s.rotation = s.gen_rotation(rng, grads);
        // Caching the rasterized stroke
        s.raster = Some(s.draw(brushes));
        s
    }

    pub fn mutate(
        &self,
        rng: &mut rand::rngs::ThreadRng,
        scale: &ScaleRange,
        img_size: (u32, u32),
        brushes: &Vec<GrayAlphaImage>,
        grads: Option<&GradientImage>,
        cdf: Option<&Distribution>,
    ) -> Stroke {
        // We only mutate few parameter of a stroke
        // as genetic algorithm does
        let mut mutations = vec![0, 1, 2, 3, 4];
        mutations.shuffle(rng);
        let nb_mutations = rng.gen_range(1, mutations.len() + 1);
        mutations.resize(nb_mutations, -1);

        // Here an important aspect: some parameter of the stroke
        // depends to some other. For example, the stroke orientation
        // and the position. It is for this reason that we sort
        // the mutation that we selected in case this happens.
        mutations.sort();

        let mut new_stroke = self.clone();
        for m in mutations {
            match m {
                0 => new_stroke.value = rng.gen_range(0.0, 1.0),
                1 => new_stroke.size = scale.value(rng.gen_range(0.0, 1.0)),
                2 => new_stroke.pos = Stroke::gen_position(rng, img_size, cdf),
                3 => new_stroke.rotation = self.gen_rotation(rng, grads),
                4 => new_stroke.brush_id = rng.gen_range(0, brushes.len()),
                _ => panic!("Not covered mutation approach"),
            };
        }
        new_stroke.raster = Some(new_stroke.draw(brushes));

        new_stroke
    }

    pub fn get_position(&self) -> (i32, i32) {
        let r = self.raster.as_ref().unwrap();
        (
            self.pos.0 - (r.width() as f32 * 0.5) as i32,
            self.pos.1 - (r.height() as f32 * 0.5) as i32,
        )
    }

    fn draw(&self, brushes: &Vec<GrayAlphaImage>) -> GrayAlphaImage {
        // Compute the affine transformation
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
        // This is to optimize the image blending and the memory storage
        // optimally, we want to do this operation at the same time
        // that we apply the affine transformation
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
        // Use the stroke color and use alpha of the bruch
        b.pixels_mut().for_each(|p| {
            p[1] = p[0];
            p[0] = (self.value * 255.0) as u8;
        });
        b
    }
}
