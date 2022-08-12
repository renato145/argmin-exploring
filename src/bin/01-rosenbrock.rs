use argmin::{
    core::{CostFunction, Executor, Gradient, Hessian},
    solver::{gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch},
};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};

#[derive(Default, Debug)]
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(-rosenbrock_2d(param, self.a, self.b))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok(rosenbrock_2d_derivative(param, self.a, self.b))
    }
}

impl Hessian for Rosenbrock {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let t = rosenbrock_2d_hessian(param, self.a, self.b);
        Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
    }
}

fn main() {
    let problem = Rosenbrock::default();
    let init_param = vec![-1.2, 1.0];
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);
    let res = Executor::new(problem, solver)
        .configure(|state| state.param(init_param).max_iters(10))
        .run()
        .unwrap();
    println!("{}", res);
    let state = res.state();
    println!("{:#?}", state);
}
