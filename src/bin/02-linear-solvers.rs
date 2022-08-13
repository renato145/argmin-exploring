use std::time::Duration;

use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        Executor, State,
    },
    solver::{
        gradientdescent::SteepestDescent,
        linesearch::{
            condition::ArmijoCondition, BacktrackingLineSearch, HagerZhangLineSearch,
            MoreThuenteLineSearch,
        },
    },
};
use argmin_exploring::Rosenbrock;
use tabled::{Style, Table, Tabled};

#[derive(Tabled)]
struct Result {
    method: String,
    best_cost: f64,
    time: String,
}

impl Result {
    fn new(method: impl ToString, best_cost: f64, time: Option<Duration>) -> Self {
        let time = time
            .map(|d| format!("{d:?}"))
            .unwrap_or_else(|| "-".to_string());
        Self {
            method: method.to_string(),
            best_cost,
            time,
        }
    }
}

fn main() {
    println!("Line solver methods");
    let problem = Rosenbrock::default();
    let init_param = vec![10.2, -20.0];
    let iterations = 10;
    let mut results = Vec::new();

    // Backtracking
    let backtracking = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001).unwrap());
    let backtracking_solver = SteepestDescent::new(backtracking);
    let backtracking_res = Executor::new(problem, backtracking_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("Backtracking: {backtracking_res}");
    results.push(Result::new(
        "Backtracking",
        backtracking_res.state.get_best_cost(),
        backtracking_res.state.get_time(),
    ));

    // More-Thuente
    let morethuente = MoreThuenteLineSearch::new();
    let morethuente_solver = SteepestDescent::new(morethuente);
    let morethuente_res = Executor::new(problem, morethuente_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("More-Thuente: {morethuente_res}");
    results.push(Result::new(
        "More-Thuente",
        morethuente_res.state.get_best_cost(),
        morethuente_res.state.get_time(),
    ));

    // Hager-Zhang
    let hagerzhang = HagerZhangLineSearch::new();
    let hagerzhang_solver = SteepestDescent::new(hagerzhang);
    let hagerzhang_res = Executor::new(problem, hagerzhang_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()
        .unwrap();
    println!("Hager-Zhang: {hagerzhang_res}");
    results.push(Result::new(
        "Hager-Zhang",
        hagerzhang_res.state.get_best_cost(),
        hagerzhang_res.state.get_time(),
    ));

    let table = Table::new(results).with(Style::modern()).to_string();
    println!("Results:\n{table}");
}
