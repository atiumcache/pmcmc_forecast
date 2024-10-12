import src.pmcmc.main as pmcmc_main
from src.utils.location_codes import location_codes
import multiprocessing as mp


def run_pmcmc(loc_code):
    pmcmc_main.main(loc_code, '2024-04-27')


if __name__ == '__main__':
    n_processes = min(mp.cpu_count() - 1, 50)
    
    with mp.Pool(processes=n_processes) as pool:
        pool.map(run_pmcmc, location_codes)




