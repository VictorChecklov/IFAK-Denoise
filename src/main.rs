use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

fn main() {
    let img = image::open("./tests/kodim20.png").unwrap().to_luma8();
    let mut image: GrayImage = img;
    let output = denoise(&mut image);

    image.save("kodim20.png").unwrap();
    match output {
        Some(output) => {
            output.save("output.png").unwrap();
        }
        None => {
            println!("No output");
        }
    }
}


fn pad_image(image: &GrayImage, pad_size: usize) -> GrayImage {
    let (width, height) = image.dimensions();
    let new_width = width + 2 * pad_size as u32;
    let new_height = height + 2 * pad_size as u32;
    let mut padded = GrayImage::new(new_width, new_height);

    for y in 0..height {
        for x in 0..width {
            padded.put_pixel(x + pad_size as u32, y + pad_size as u32, *image.get_pixel(x, y));
        }
    }

    for y in 0..pad_size as u32 {
        for x in 0..new_width {
            let src_y = if y < pad_size as u32 { pad_size as u32 - 1 - y } else { y - pad_size as u32 };
            let src_x = x;

            padded.put_pixel(x, y, *padded.get_pixel(src_x, src_y + pad_size as u32));
            padded.put_pixel(x, new_height - 1 - y, *padded.get_pixel(src_x, new_height - 2 * pad_size as u32 - 1 - src_y));
        }
    }

    for y in 0..new_height {
        for x in 0..pad_size as u32 {
            let src_x = if x < pad_size as u32 { pad_size as u32 - 1 - x } else { x - pad_size as u32 };
            let src_y = y;
            padded.put_pixel(x, y, *padded.get_pixel(src_x + pad_size as u32, src_y));
            padded.put_pixel(new_width - 1 - x, y, *padded.get_pixel(new_width - 2 * pad_size as u32 - 1 - src_x, src_y));
        }
    }

    padded
}

fn apply_filter<F>(
    image: &GrayImage,
    kernel_height: usize,
    kernel_width: usize,
    filter: F,
) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>>
where
    F: Fn((&[&[u8]], usize, usize)) -> f64 + Sync + Send,
{
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    if kernel_height == 0 || kernel_width == 0 || kernel_height > height || kernel_width > width {
        return None;
    }

    let pad_size = (kernel_height - 1) / 2;
    let padded_image = pad_image(image, pad_size);
    let (padded_width, _) = padded_image.dimensions();
    let padded_width = padded_width as usize;
    let mut output = vec![0; width * height];

    output.par_iter_mut().enumerate().for_each(|(index, pixel)| {
        let row_index = index / width;
        let col_index = index % width;

        let center_y = row_index + pad_size;
        let center_x = col_index + pad_size;

        let window: Vec<&[u8]> = (center_y - pad_size..center_y - pad_size + kernel_height)
            .map(|y| {
                let row_start = y * padded_width + (center_x - pad_size);
                let row_end = row_start + kernel_width;
                &padded_image.as_raw()[row_start..row_end]
            })
            .collect();

        *pixel = filter((&window, center_y, center_x)).clamp(0.0, 255.0) as u8;
    });

    GrayImage::from_raw(width as u32, height as u32, output)
}

pub fn denoise(image: &mut GrayImage) -> Option<ImageBuffer<Luma<u8>, Vec<u8>>> {
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    // STAGE 1: Intuitionistic Fuzzification
    // Get the intuitionistic fuzzy set
    let mut intuitionistic_fuzzy_mat = [[0f64; 3]; 256]; // [gray_value, membership, non-membership]
    let mut energy_mat = [0f64; 256];
    image.iter().for_each(|&value| energy_mat[value as usize] += 1.0);

    intuitionistic_fuzzy_mat.par_iter_mut().enumerate().for_each(|(layer, output)| {
        let layer_f64 = layer as f64;
        let (background_sum, background_weighted_sum) = (0..=layer)
            .map(|p| (energy_mat[p], p as f64 * energy_mat[p]))
            .fold((0.0, 0.0), |(sum, w_sum), (e, w)| (sum + e, w_sum + w));
        let (foreground_sum, foreground_weighted_sum) = ((layer + 1)..256)
            .map(|p| (energy_mat[p], p as f64 * energy_mat[p]))
            .fold((0.0, 0.0), |(sum, w_sum), (e, w)| (sum + e, w_sum + w));

        let background_average_gray = if background_sum > 0.0 {
            background_weighted_sum / background_sum
        } else {
            0.0
        };
        let foreground_average_gray = if foreground_sum > 0.0 {
            foreground_weighted_sum / foreground_sum
        } else {
            255.0
        };

        let membership_reference =
            1.0 - 0.5 * ((layer_f64 / 255.0) - (background_average_gray / 255.0)).powi(2);
        let non_membership_reference =
            1.0 - 0.5 * ((layer_f64 / 255.0) - (foreground_average_gray / 255.0)).powi(2);

        let intuitive_index = 1.0 - membership_reference.max(non_membership_reference);

        if membership_reference >= non_membership_reference {
            *output = [
                layer_f64,
                membership_reference,
                1.0 - membership_reference - intuitive_index,
            ];
        } else {
            *output = [
                layer_f64,
                1.0 - non_membership_reference - intuitive_index,
                non_membership_reference,
            ];
        }
    });

    // Stage 2: Get the Best Segmentation Threshold
    // To differentiate the background and the foreground
    let segmentation_threshold = intuitionistic_fuzzy_mat
        .par_iter()
        .enumerate()
        .map(|(l, layer)| {
            let knowledge = (layer[1] + layer[2]) / (1.0 + layer[1].min(layer[2]));
            (l as u8, knowledge)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(l, _)| l)
        .unwrap_or(0);

    // Stage 3: Get the Noise Possibility Matrix
    // Define the possibility that a pixel belongs to noise
    let (threshold_background_sum, threshold_background_weighted_sum) =
        (0..=segmentation_threshold as usize)
        .map(|p| (energy_mat[p], p as f64 * energy_mat[p]))
        .fold((0.0, 0.0), |(sum, w_sum), (e, w)| (sum + e, w_sum + w));

    let (threshold_foreground_sum, threshold_foreground_weighted_sum) =
        ((segmentation_threshold as usize + 1)..=255)
        .map(|p| (energy_mat[p], p as f64 * energy_mat[p]))
        .fold((0.0, 0.0), |(sum, w_sum), (e, w)| (sum + e, w_sum + w));

    let threshold_background_average = if threshold_background_sum > 0.0 {
        threshold_background_weighted_sum / threshold_background_sum
    } else {
        0.0
    };
    let threshold_foreground_average = if threshold_foreground_sum > 0.0 {
        threshold_foreground_weighted_sum / threshold_foreground_sum
    } else {
        255.0
    };

    let noise_possibility_mat: Vec<f64> = image
        .iter()
        .map(|&pixel| {
            let pixel_f = pixel as f64;
            if (pixel_f - threshold_foreground_average).abs() < threshold_foreground_average {
                0.0
            } else if 2.0 * threshold_foreground_average <= pixel_f && pixel_f <= threshold_background_average {
                (pixel_f - 2.0 * threshold_foreground_average) /
                    (2.0 * (threshold_background_average - threshold_foreground_average))
            } else {
                1.0
            }
        })
        .collect();

    let knowledge_mat: Vec<f64> = image
        .iter()
        .map(|&pixel| {
            let layer = &intuitionistic_fuzzy_mat[pixel as usize];
            (layer[1] + layer[2]) / (1.0 + layer[1].min(layer[2]))
        })
        .collect();

    // STAGE 4: Final Operations
    // Use a filter to denoise
    let filter = |(region, center_y, center_x): (&[&[u8]], usize, usize)| {
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..7 {
            for j in 0..7 {
                if (i, j) != (3, 3) {
                    let pixel_y = (center_y as i32 - 3 + i as i32) as usize;
                    let pixel_x = (center_x as i32 - 3 + j as i32) as usize;
                    let pixel_idx = (pixel_y - 3) * width + (pixel_x - 3);

                    if pixel_idx < width * height {
                        let noise_possibility = noise_possibility_mat[pixel_idx];
                        let knowledge = knowledge_mat[pixel_idx];
                        let weight = knowledge * (1.0 - noise_possibility);
                        numerator += weight * region[i][j] as f64;
                        denominator += weight;
                    }
                }
            }
        }

        if denominator > 0.0 {
            (numerator / denominator).clamp(0.0, 255.0)
        } else {
            region[3][3] as f64
        }
    };
    apply_filter(image, 7, 7, filter)
}