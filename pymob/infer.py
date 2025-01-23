import os
import sys
import click
import resource

from pymob.utils import help
from pymob.utils.store_file import prepare_casestudy, import_package
from pymob.simulation import SimulationBase
from pymob.sim.config import Config

@click.command()
@click.option("-c", "--case_study", type=str, default="lotka_volterra_case_study", 
              help=help.case_study)
@click.option("-s", "--scenario", type=str, default="test_scenario", 
              help=help.scenario)
@click.option("-p", "--package", type=str, default="case_studies", 
              help=help.package)
@click.option("-o", "--output", type=str, default=None, 
              help="Set `case-study.output` directory")
@click.option("-r", "--random_seed", type=int, default=None, 
              help="Set `simulation.seed`")
@click.option("-n", "--n_cores", type=int, default=None, 
              help="The number of cores to be used for multiprocessing")
@click.option("--inference_backend", type=str, default="pymoo")
@click.option("--only_ppc", type=bool, default=False)
def main(case_study, scenario, package, output, random_seed, n_cores, inference_backend, only_ppc):
    
    cfg = os.path.join(package, case_study, "scenarios", scenario, "settings.cfg")
    config = Config(cfg)
    config.case_study.name = case_study
    config.case_study.scenario = scenario
    config.case_study.package = package
    config.import_casestudy_modules()

    if n_cores is not None: config.multiprocessing.cores = n_cores
    if random_seed is not None: config.simulation.seed = random_seed
    if output is not None: config.case_study.output = output

    # import simulation      
    Simulation = config.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()
    sim.config.save(os.path.join(sim.output_path, "settings.cfg"), force=True)

    sim.set_inferer(backend=inference_backend)
    if not only_ppc:
        sim.prior_predictive_checks()
        sim.inferer.run()
        sim.inferer.store_results()
    else:
        sim.inferer.load_results()

    sim.posterior_predictive_checks()
    sim.inferer.plot()

    max_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    print("RESOURCE USAGE")
    print("==============")
    print(f"Max RSS: {max_ram_mb} M")


if __name__ == "__main__":
    main()