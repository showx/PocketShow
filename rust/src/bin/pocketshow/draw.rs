use super::track::Person;

pub fn frame_buffer(rgb: &[u8], width: u32, height: u32, people: &[Person], locked: Option<i64>) -> Vec<u32> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for pixel in rgb.chunks_exact(3) {
        pixels.push((u32::from(pixel[0]) << 16) | (u32::from(pixel[1]) << 8) | u32::from(pixel[2]));
    }
    for person in people {
        let color = if Some(person.id) == locked { 0x003DDC97 } else { 0x00F2F2F2 };
        rect(&mut pixels, width, height, person.x1, person.y1, person.x2, person.y2, color);
        if let Some(face) = person.face {
            rect(
                &mut pixels,
                width,
                height,
                face[0],
                face[1],
                face[0] + face[2],
                face[1] + face[3],
                0x00F5C542,
            );
        }
    }
    pixels
}

fn rect(pixels: &mut [u32], width: u32, height: u32, x1: f32, y1: f32, x2: f32, y2: f32, color: u32) {
    let x1 = x1.round().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let x2 = x2.round().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y1 = y1.round().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let y2 = y2.round().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let (x1, x2) = (x1.min(x2), x1.max(x2));
    let (y1, y2) = (y1.min(y2), y1.max(y2));
    for x in x1..=x2 {
        put(pixels, width, x, y1, color);
        put(pixels, width, x, y2, color);
    }
    for y in y1..=y2 {
        put(pixels, width, x1, y, color);
        put(pixels, width, x2, y, color);
    }
}

fn put(pixels: &mut [u32], width: u32, x: u32, y: u32, color: u32) {
    let index = (y * width + x) as usize;
    if index < pixels.len() {
        pixels[index] = color;
    }
}
