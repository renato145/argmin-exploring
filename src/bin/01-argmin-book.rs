use argmin::{
    core::{
        checkpointing::{CheckpointingFrequency, FileCheckpoint},
        observers::{ObserverMode, SlogLogger},
        Executor,
    },
    solver::{gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch},
};
use argmin_exploring::Rosenbrock;
use ndarray::array;

fn main() {
    let max_iters = std::env::args()
        .nth(1)
        .map(|x| {
            x.parse()
                .unwrap_or_else(|x| panic!("Invalid number for `max_iters`: {x}"))
        })
        .unwrap_or(10);

    let problem = Rosenbrock::default();
    let init_param = array![10.2, -20.0];
    let linesearch = MoreThuenteLineSearch::new();
    let solver = SteepestDescent::new(linesearch);
    let checkpoint = FileCheckpoint::new(
        "checkpoints",
        "01-argmin-book",
        CheckpointingFrequency::Every(5),
    );

    let res = Executor::new(problem, solver)
        .configure(|state| state.param(init_param).max_iters(max_iters))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .checkpointing(checkpoint)
        .run()
        .unwrap();
    println!("{}", res);
    let state = res.state();
    println!("{:?}", state);
}
