import click
import resource

from pymob.utils import help
from pymob.utils.store_file import prepare_casestudy, import_package
from pymob.simulation import SimulationBase

@click.command()
@click.option("-c", "--case_study", type=str, default="test_case_study", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="test_scenario", 
              help=help.scenario)
@click.option("-p", "--package", type=str, default="case_studies", 
              help=help.package)
@click.option("-n", "--n_cores", type=int, default=2, 
              help="The number of cores to be used for multiprocessing")
@click.option("--inference_backend", type=str, default="pymoo")
def main(case_study, scenario, package, inference_backend, n_cores):
    
    config = prepare_casestudy(
        case_study=(case_study, scenario), 
        config_file="settings.cfg", 
        pkg_dir=package
    )

    if n_cores is not None: config.set("multiprocessing", "cores", str(n_cores))
    
    # import package        
    pkg = import_package(package_path=config["case-study"]["package"])
    Simulation: SimulationBase = getattr(
        pkg.sim, config["case-study"].get("simulation", fallback="Simulation"))
    sim = Simulation(config)

    sim.set_inferer(backend=inference_backend)
    sim.prior_predictive_checks()
    sim.inferer.run()
    sim.inferer.store_results()
    sim.posterior_predictive_checks()
    sim.inferer.plot()

    max_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    print("RESOURCE USAGE")
    print("==============")
    print(f"Max RSS: {max_ram_mb} M")


if __name__ == "__main__":
    main()