use argmin::core::{CostFunction, Gradient, Hessian};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use ndarray::{Array1, Array2};

/// The rosenbrock function is defined as:
/// $ f(x,y) = (a-x)^2 + b(y-x^2)^2 $
#[derive(Debug, Clone, Copy)]
pub struct RosenbrockND {
    a: f64,
    b: f64,
}

impl RosenbrockND {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl Default for RosenbrockND {
    fn default() -> Self {
        Self::new(1.0, 100.0)
    }
}

impl CostFunction for RosenbrockND {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(rosenbrock_2d(&param.to_vec(), self.a, self.b))
    }
}

impl Gradient for RosenbrockND {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let gradient = rosenbrock_2d_derivative(&param.to_vec(), self.a, self.b);
        Ok(Array1::from_vec(gradient))
    }
}

impl Hessian for RosenbrockND {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let h = rosenbrock_2d_hessian(&param.to_vec(), self.a, self.b);
        Ok(Array2::from_shape_vec((2, 2), h)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rosenbrock() {
        let f = RosenbrockND::default();
        let params = vec![
            array![10.0, 5.0],
            array![5.0, 2.0],
            array![0.0, 1.0],
            array![-4.0, 0.0],
            array![-10.0, -2.0],
        ];
        for param in params {
            let cost = f.cost(&param).unwrap();
            let gradient = f.gradient(&param).unwrap();
            let hessian = f.hessian(&param).unwrap();
            println!("With params {param:?}:");
            println!("\tcost: {cost}");
            println!("\tgradient: {gradient:?}");
            println!("\thessian: {hessian:?}");
        }
    }
}
