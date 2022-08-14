use argmin::{
    core::{
        observers::{ObserverMode, SlogLogger},
        Executor, State, TerminationReason,
    },
    solver::{
        conjugategradient::{beta::PolakRibiere, NonlinearConjugateGradient},
        gradientdescent::SteepestDescent,
        linesearch::{
            condition::ArmijoCondition, BacktrackingLineSearch, HagerZhangLineSearch,
            MoreThuenteLineSearch,
        },
        newton::{Newton, NewtonCG},
        quasinewton::{SR1TrustRegion, BFGS, DFP, LBFGS},
        trustregion::{CauchyPoint, Steihaug, TrustRegion},
    },
};
use argmin_exploring::Rosenbrock;
use ndarray::{array, Array2};
use std::time::Duration;
use tabled::{Style, Table, Tabled};

#[derive(Tabled)]
#[tabled(rename_all = "Pascal")]
struct Result {
    family: String,
    method: String,
    best_cost: f64,
    time: String,
    termination_reason: TerminationReason,
}

impl Result {
    fn new(
        family: impl ToString,
        method: impl ToString,
        best_cost: f64,
        time: Option<Duration>,
        termination_reason: TerminationReason,
    ) -> Self {
        let time = time
            .map(|d| format!("{d:?}"))
            .unwrap_or_else(|| "-".to_string());
        Self {
            family: family.to_string(),
            method: method.to_string(),
            best_cost,
            time,
            termination_reason,
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let iterations = args
        .next()
        .map(|x| {
            x.parse()
                .unwrap_or_else(|x| panic!("Invalid number for `max_iters`: {x}"))
        })
        .unwrap_or(100);
    let log_every = args
        .next()
        .map(|x| {
            x.parse()
                .unwrap_or_else(|x| panic!("Invalid number for `log_every`: {x}"))
        })
        .unwrap_or(10);

    println!("Line solver methods");
    let problem = Rosenbrock::default();
    let init_param = array![10.2, -20.0];
    let mut results = Vec::new();

    // Linear search - Backtracking
    let backtracking = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001).unwrap());
    let backtracking_solver = SteepestDescent::new(backtracking);
    let backtracking_res = Executor::new(problem, backtracking_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("Backtracking: {backtracking_res}");
    results.push(Result::new(
        "Linear search",
        "Backtracking",
        backtracking_res.state.get_best_cost(),
        backtracking_res.state.get_time(),
        backtracking_res.state.get_termination_reason(),
    ));

    // Linear search - More-Thuente
    let morethuente = MoreThuenteLineSearch::new();
    let morethuente_solver = SteepestDescent::new(morethuente);
    let morethuente_res = Executor::new(problem, morethuente_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("More-Thuente: {morethuente_res}");
    results.push(Result::new(
        "Linear search",
        "More-Thuente",
        morethuente_res.state.get_best_cost(),
        morethuente_res.state.get_time(),
        morethuente_res.state.get_termination_reason(),
    ));

    // Linear search - Hager-Zhang
    let hagerzhang = HagerZhangLineSearch::new();
    let hagerzhang_solver = SteepestDescent::new(hagerzhang);
    let hagerzhang_res = Executor::new(problem, hagerzhang_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("Hager-Zhang: {hagerzhang_res}");
    results.push(Result::new(
        "Linear search",
        "Hager-Zhang",
        hagerzhang_res.state.get_best_cost(),
        hagerzhang_res.state.get_time(),
        hagerzhang_res.state.get_termination_reason(),
    ));

    // Trust Region - Cauchy Point
    let cauchy_point = CauchyPoint::new();
    let cauchy_point_solver = TrustRegion::new(cauchy_point);
    let cauchy_point_res = Executor::new(problem, cauchy_point_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("Cauchy-Point: {cauchy_point_res}");
    results.push(Result::new(
        "Trust region",
        "Cauchy-Point",
        cauchy_point_res.state.get_best_cost(),
        cauchy_point_res.state.get_time(),
        cauchy_point_res.state.get_termination_reason(),
    ));

    // BUG HERE: https://github.com/argmin-rs/argmin/issues/246
    // // Trust Region - Dogleg
    // let dogleg = Dogleg::new();
    // let dogleg_solver = TrustRegion::new(dogleg);
    // let dogleg_res = Executor::new(problem, dogleg_solver)
    //     .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
    //     .configure(|state| state.param(init_param).max_iters(iterations))
    //     .run()
    //     .unwrap();
    // println!("Dogleg: {dogleg_res}");
    // results.push(Result::new(
    //     "Trust region",
    //     "Dogleg",
    //     dogleg_res.state.get_best_cost(),
    //     dogleg_res.state.get_time(),
    //     dogleg_res.state.get_termination_reason(),
    // ));

    // Trust Region - Steighaug
    let steighaug = Steihaug::new();
    let steighaug_solver = TrustRegion::new(steighaug);
    let steighaug_res = Executor::new(problem, steighaug_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("steighaug: {steighaug_res}");
    results.push(Result::new(
        "Trust region",
        "Steighaug",
        steighaug_res.state.get_best_cost(),
        steighaug_res.state.get_time(),
        steighaug_res.state.get_termination_reason(),
    ));

    // Conjugate Gradient - Non-linear Conjugate Gradient
    let linesearch = MoreThuenteLineSearch::new();
    let beta_method = PolakRibiere::new();
    let nlcg_solver = NonlinearConjugateGradient::new(linesearch, beta_method)
        .restart_iters(10)
        .restart_orthogonality(0.1);
    let nlcg_res = Executor::new(problem, nlcg_solver)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("non-linear conjugate gradient: {nlcg_res}");
    results.push(Result::new(
        "Conjugate Gradient",
        "Non-linear Conjugate Gradient",
        nlcg_res.state.get_best_cost(),
        nlcg_res.state.get_time(),
        nlcg_res.state.get_termination_reason(),
    ));

    // Newton - Newton's method
    let newton = Newton::new();
    let newton_res = Executor::new(problem, newton)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("newton: {newton_res}");
    results.push(Result::new(
        "Newton methods",
        "Newton",
        newton_res.state.get_best_cost(),
        newton_res.state.get_time(),
        newton_res.state.get_termination_reason(),
    ));

    // Newton - Newton-CG method
    let linesearch = MoreThuenteLineSearch::new();
    let newton_cg = NewtonCG::new(linesearch);
    let newton_cg_res = Executor::new(problem, newton_cg)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("newton_cg: {newton_cg_res}");
    results.push(Result::new(
        "Newton methods",
        "Newton-CG",
        newton_cg_res.state.get_best_cost(),
        newton_cg_res.state.get_time(),
        newton_cg_res.state.get_termination_reason(),
    ));

    // Quasi Newton - BFGS
    let linesearch = MoreThuenteLineSearch::new();
    let bfgs = BFGS::new(linesearch);
    let bfgs_res = Executor::new(problem, bfgs)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| {
            state
                .param(init_param.clone())
                // Hessian type required to initialize
                .inv_hessian(Array2::eye(2))
                .max_iters(iterations)
        })
        .run()
        .unwrap();
    println!("bfgs: {bfgs_res}");
    results.push(Result::new(
        "Quasi-Newton methods",
        "BFGS",
        bfgs_res.state.get_best_cost(),
        bfgs_res.state.get_time(),
        bfgs_res.state.get_termination_reason(),
    ));

    // Quasi Newton - DFP
    let linesearch = MoreThuenteLineSearch::new();
    let dfp = DFP::new(linesearch);
    let dfp_res = Executor::new(problem, dfp)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| {
            state
                .param(init_param.clone())
                // Hessian type required to initialize
                .inv_hessian(Array2::eye(2))
                .max_iters(iterations)
        })
        .run()
        .unwrap();
    println!("dfp: {dfp_res}");
    results.push(Result::new(
        "Quasi-Newton methods",
        "DFP",
        dfp_res.state.get_best_cost(),
        dfp_res.state.get_time(),
        dfp_res.state.get_termination_reason(),
    ));

    // Quasi Newton - L-BFGS
    let linesearch = MoreThuenteLineSearch::new();
    let lbfgs = LBFGS::new(linesearch, 5);
    let lbfgs_res = Executor::new(problem, lbfgs)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("lbfgs: {lbfgs_res}");
    results.push(Result::new(
        "Quasi-Newton methods",
        "L-BFGS",
        lbfgs_res.state.get_best_cost(),
        lbfgs_res.state.get_time(),
        lbfgs_res.state.get_termination_reason(),
    ));

    // Quasi Newton - SR1: https://github.com/argmin-rs/argmin/issues/221

    // Quasi Newton - SR1-Trust Region
    let subproblem = Steihaug::new();
    let sr1tr = SR1TrustRegion::new(subproblem);
    let sr1tr_res = Executor::new(problem, sr1tr)
        .add_observer(SlogLogger::term(), ObserverMode::Every(log_every))
        .configure(|state| state.param(init_param.clone()).max_iters(iterations))
        .run()
        .unwrap();
    println!("sr1tr: {sr1tr_res}");
    results.push(Result::new(
        "Quasi-Newton methods",
        "SR1-TrustRegion",
        sr1tr_res.state.get_best_cost(),
        sr1tr_res.state.get_time(),
        sr1tr_res.state.get_termination_reason(),
    ));

    // Results table
    let table = Table::new(results).with(Style::modern()).to_string();
    println!("Results using {iterations} iterations:\n{table}");
}
