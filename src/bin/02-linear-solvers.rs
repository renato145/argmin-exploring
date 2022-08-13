use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        Executor,
    },
    solver::{
        gradientdescent::SteepestDescent,
        linesearch::{condition::ArmijoCondition, BacktrackingLineSearch},
    },
};
use argmin_exploring::Rosenbrock;

fn main() {
    println!("Line solver methods");
    let problem = Rosenbrock::default();
    let init_param = vec![10.2, -20.0];
    let iterations = 10;

    // Backtracking
    let backtracking = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001).unwrap());
    let backtracking_solver = SteepestDescent::new(backtracking);
    let backtracking_res = Executor::new(problem, backtracking_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()
        .unwrap();
    println!("Backtracking: {backtracking_res}");
}
