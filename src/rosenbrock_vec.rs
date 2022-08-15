use argmin::core::{CostFunction, Gradient, Hessian};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};

/// The rosenbrock function is defined as:
/// $ f(x,y) = (a-x)^2 + b(y-x^2)^2 $
#[derive(Debug, Clone, Copy)]
pub struct RosenbrockVec {
    a: f64,
    b: f64,
}

impl RosenbrockVec {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl Default for RosenbrockVec {
    fn default() -> Self {
        Self::new(1.0, 100.0)
    }
}

impl CostFunction for RosenbrockVec {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(rosenbrock_2d(&param.to_vec(), self.a, self.b))
    }
}

impl Gradient for RosenbrockVec {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok(rosenbrock_2d_derivative(&param.to_vec(), self.a, self.b))
    }
}

impl Hessian for RosenbrockVec {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let t = rosenbrock_2d_hessian(&param.to_vec(), self.a, self.b);
        Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock() {
        let f = RosenbrockVec::default();
        let params = vec![
            vec![10.0, 5.0],
            vec![5.0, 2.0],
            vec![0.0, 1.0],
            vec![-4.0, 0.0],
            vec![-10.0, -2.0],
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
