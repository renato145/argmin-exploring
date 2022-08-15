use std::sync::{Arc, Mutex};

use argmin::{
    core::{CostFunction, Gradient, Hessian},
    solver::simulatedannealing::Anneal,
};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use ndarray::{array, Array1, Array2};
use rand::{distributions::Uniform, Rng};
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

/// The rosenbrock function is defined as:
/// $ f(x,y) = (a-x)^2 + b(y-x^2)^2 $
#[derive(Debug, Clone)]
pub struct RosenbrockND {
    a: f64,
    b: f64,
    lower_bound: Array1<f64>,
    upper_bound: Array1<f64>,
    /// Random number generator. We use a `Arc<Mutex<_>>` here because `ArgminOperator` requires
    /// `self` to be passed as an immutable reference. This gives us thread safe interior
    /// mutability.
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl RosenbrockND {
    pub fn new(a: f64, b: f64, lower_bound: Array1<f64>, upper_bound: Array1<f64>) -> Self {
        Self {
            a,
            b,
            lower_bound,
            upper_bound,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
        }
    }
}

impl Default for RosenbrockND {
    fn default() -> Self {
        Self::new(1.0, 100.0, array![-5.0, -5.0], array![5.0, 5.0])
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

impl Anneal for RosenbrockND {
    type Param = Array1<f64>;
    type Output = Array1<f64>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Self::Param,
        temp: Self::Float,
    ) -> Result<Self::Output, argmin::core::Error> {
        let mut param_n = param.clone();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::from(0..param.len());
        // Perform modifications to a degree proportional to the current temperature `temp`.
        for _ in 0..(temp.floor() as u64 + 1) {
            // Compute random index of the parameter vector using the supplied random number
            // generator.
            let idx = rng.sample(distr);

            // Compute random number in [0.1, 0.1].
            let val = rng.sample(Uniform::new_inclusive(-0.1, 0.1));

            // modify previous parameter value at random position `idx` by `val`
            param_n[idx] += val;

            // check if bounds are violated. If yes, project onto bound.
            param_n[idx] = param_n[idx].clamp(self.lower_bound[idx], self.upper_bound[idx]);
        }
        Ok(param_n)
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
