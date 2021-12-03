const UNITS: i16 = 170_i16;
const SCALE: i16 = 64;
const MIN: i16 = 0;
const MAX: i16 = SCALE;

#[derive(Debug, Clone)]
pub struct Psqt<'a, const INPUT: usize, const OUTPUT: usize> {
    weights: &'a [[i32; OUTPUT]; INPUT],
    out: [i32; OUTPUT],
}

impl<'a, const INPUT: usize, const OUTPUT: usize> Psqt<'a, INPUT, OUTPUT> {
    pub fn new(weights: &'a [[i32; OUTPUT]; INPUT]) -> Self {
        Self {
            weights,
            out: [0_i32; OUTPUT],
        }
    }

    #[inline]
    pub fn incr_ff<const CHANGE: i32>(&mut self, index: usize) {
        for (out, &weight) in self.out.iter_mut().zip(&self.weights[index]) {
            *out += weight * CHANGE;
        }
    }

    pub fn get(&self) -> &[i32; OUTPUT] {
        &self.out
    }
}

#[derive(Debug, Clone)]
pub struct Incremental<'a, const INPUT: usize, const OUTPUT: usize> {
    weights: &'a [[i8; OUTPUT]; INPUT],
    out: [i16; OUTPUT],
}

impl<'a, const INPUT: usize, const OUTPUT: usize> Incremental<'a, INPUT, OUTPUT> {
    pub fn new(weights: &'a [[i8; OUTPUT]; INPUT], bias: [i16; OUTPUT]) -> Self {
        Self { weights, out: bias }
    }

    #[inline]
    pub fn incr_ff<const CHANGE: i8>(&mut self, index: usize) {
        for (out, &weight) in self.out.iter_mut().zip(&self.weights[index]) {
            *out += (weight * CHANGE) as i16;
        }
    }

    pub fn get(&self) -> &[i16; OUTPUT] {
        &self.out
    }
}

#[derive(Debug, Clone)]
pub struct Dense<'a, const INPUT: usize, const OUTPUT: usize> {
    weights: &'a [[i32; OUTPUT]; INPUT],
    bias: [i32; OUTPUT],
}

impl<'a, const INPUT: usize, const OUTPUT: usize> Dense<'a, INPUT, OUTPUT> {
    pub fn new(weights: &'a [[i32; OUTPUT]; INPUT], bias: [i16; OUTPUT]) -> Self {
        Self {
            weights,
            bias: i16_to_i32(bias),
        }
    }

    #[inline]
    pub fn ff_sym(
        &self,
        w_inputs: &[i8; INPUT],
        b_inputs: &[i8; INPUT],
        bucket: usize,
    ) -> [i32; OUTPUT] {
        let mut w_out = self.bias;
        let mut b_out = self.bias;
        for ((&w_input, &b_input), weights) in
            w_inputs.iter().zip(b_inputs.iter()).zip(&*self.weights)
        {
            for ((w_out, b_out), &weight) in w_out[bucket..bucket + 1]
                .iter_mut()
                .zip(b_out[bucket..bucket + 1].iter_mut())
                .zip(weights[bucket..bucket + 1].iter())
            {
                *w_out += weight * w_input as i32;
                *b_out += weight * b_input as i32;
            }
        }
        let mut out = [0_i32; OUTPUT];
        for (out, (w_out, b_out)) in out.iter_mut().zip(w_out.iter_mut().zip(b_out.iter_mut())) {
            *out = (*w_out - *b_out) / 2;
        }
        out
    }
}

#[inline]
pub fn out(x: i32) -> i16 {
    (x as f32 * UNITS as f32 / (SCALE * SCALE) as f32) as i16
}

#[inline]
const fn i16_to_i32<const N: usize>(array: [i16; N]) -> [i32; N] {
    let mut out = [0_i32; N];
    let mut index = 0;
    while index < N {
        out[index] = array[index] as i32;
        index += 1;
    }
    out
}

#[inline]
pub fn clipped_relu<const N: usize>(array: [i16; N]) -> [i8; N] {
    let mut out = [0_i8; N];
    for (&x, clipped) in array.iter().zip(out.iter_mut()) {
        *clipped = x.max(MIN).min(MAX) as i8;
    }
    out
}
