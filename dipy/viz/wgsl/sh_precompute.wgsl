{$ include 'dipy.utils.wgsl' $}

const L_MAX = i32({{ l_max }});
const NUM_COEFFS = i32({{ n_coeffs }});
const PHI_SAMPLES = 64u;
const THETA_SAMPLES = 32u;
const TOTAL_SAMPLES = PHI_SAMPLES * THETA_SAMPLES;

@group(0) @binding(0) var<storage, read_write> precomputed_sh_basis: array<f32>;
@group(0) @binding(1) var<storage, read_write> precomputed_normals: array<f32>;
@group(0) @binding(2) var<storage, read_write> precomputed_legendre: array<f32>;

fn factorial_ratio(l: i32, m: i32) -> f32 {
    if (m == 0 || l == 0) {
        return 1.0;
    }
    var result = 1.0;
    let start = l - m + 1;
    let stop = l + m;
    if (stop < start) {
        return 1.0;
    }
    for (var k: i32 = start; k <= stop; k++) {
        result /= f32(k);
    }
    return result;
}

fn compute_legendre_polynomial(l: i32, m: i32, cos_theta: f32) -> f32 {
    let abs_m = abs(m);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    var pmm = 1.0;
    if (abs_m > 0) {
        var fact = 1.0;
        for (var i: i32 = 1; i <= abs_m; i++) {
            pmm *= -fact * sin_theta;
            fact += 2.0;
        }
    }

    if (l == abs_m) {
        return pmm;
    }

    let pmmp1 = cos_theta * (2.0 * f32(abs_m) + 1.0) * pmm;
    if (l == abs_m + 1) {
        return pmmp1;
    }

    var prev_prev = pmm;
    var prev = pmmp1;

    for (var curr_l: i32 = abs_m + 2; curr_l <= l; curr_l++) {
        let curr = ((2.0 * f32(curr_l) - 1.0) * cos_theta * prev -
                   (f32(curr_l) + f32(abs_m) - 1.0) * prev_prev) /
                   (f32(curr_l) - f32(abs_m));
        prev_prev = prev;
        prev = curr;
    }

    return prev;
}

fn compute_spherical_harmonic(l: i32, m: i32, theta: f32, phi: f32) -> f32 {
    let abs_m = abs(m);
    let cos_theta = cos(theta);

    let ratio = factorial_ratio(l, abs_m);
    let norm = sqrt(((2.0 * f32(l) + 1.0) / (4.0 * PI)) * ratio);

    let legendre = compute_legendre_polynomial(l, abs_m, cos_theta);

    var sh_value = norm * legendre;

    if (abs_m > 0) {
        if (m > 0) {
            sh_value *= SQRT_2 * cos(f32(abs_m) * phi);
        } else {
            sh_value *= SQRT_2 * sin(f32(abs_m) * phi);
        }
    }

    return sh_value;
}

fn compute_sh_gradient_magnitude(theta: f32, phi: f32) -> f32 {
    var gradient_sum = 0.0;
    var coeff_idx = 0;

    for (var l: i32 = 0; l <= L_MAX; l++) {
        for (var m: i32 = -l; m <= l; m++) {
            if (coeff_idx >= NUM_COEFFS) {
                break;
            }

            let sh_val = compute_spherical_harmonic(l, m, theta, phi);
            let theta_deriv = compute_spherical_harmonic(l, m, theta + 0.01, phi) - sh_val;
            let phi_deriv = compute_spherical_harmonic(l, m, theta, phi + 0.01) - sh_val;

            gradient_sum += theta_deriv * theta_deriv + phi_deriv * phi_deriv;
            coeff_idx++;
        }
    }

    return sqrt(gradient_sum) + 1e-6;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let phi_idx = global_id.x;
    let theta_idx = global_id.y;

    if (phi_idx >= PHI_SAMPLES || theta_idx >= THETA_SAMPLES) {
        return;
    }

    let phi = (f32(phi_idx) + 0.5) * 2.0 * PI / f32(PHI_SAMPLES) - PI;
    let theta = (f32(theta_idx) + 0.5) * PI / f32(THETA_SAMPLES);

    let sample_idx = theta_idx * PHI_SAMPLES + phi_idx;

    var coeff_idx = 0;
    for (var l: i32 = 0; l <= L_MAX; l++) {
        for (var m: i32 = -l; m <= l; m++) {
            if (coeff_idx >= NUM_COEFFS) {
                break;
            }

            let sh_value = compute_spherical_harmonic(l, m, theta, phi);
            let storage_idx = sample_idx * u32(NUM_COEFFS) + u32(coeff_idx);

            if (storage_idx < arrayLength(&precomputed_sh_basis)) {
                precomputed_sh_basis[storage_idx] = sh_value;
            }

            coeff_idx++;
        }
    }

    let gradient_magnitude = compute_sh_gradient_magnitude(theta, phi);
    if (sample_idx < arrayLength(&precomputed_normals)) {
        precomputed_normals[sample_idx] = gradient_magnitude;
    }

    var legendre_idx = 0u;
    for (var l: i32 = 0; l <= L_MAX; l++) {
        for (var m: i32 = 0; m <= l; m++) {
            let cos_theta = cos(theta);
            let legendre_val = compute_legendre_polynomial(l, m, cos_theta);
            let storage_idx = sample_idx * u32((L_MAX + 1) * (L_MAX + 2) / 2) + legendre_idx;

            if (storage_idx < arrayLength(&precomputed_legendre)) {
                precomputed_legendre[storage_idx] = legendre_val;
            }

            legendre_idx++;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn optimize_cache_layout(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&precomputed_sh_basis)) {
        return;
    }

}
