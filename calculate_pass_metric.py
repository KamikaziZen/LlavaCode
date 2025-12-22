import argparse
import os
import re
import math


def get_last_nonempty_line(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        nonempty_lines = [line.strip() for line in lines if line.strip()]
        return nonempty_lines[-1] if nonempty_lines else None


def parse_test_results(line):
    results = {
        'passed': 0,
        'failed': 0,
        'xfailed': 0,
        'subtests': 0,
        'warnings': 0,
        'errors': 0,
    }
    patterns = {
        'passed': r'(\d+)\s+passed',
        'failed': r'(\d+)\s+failed',
        'xfailed': r'(\d+)\s+xfailed',
        'subtests': r'(\d+)\s+subtests\s+passed',
        'warnings': r'(\d+)\s+warnings',
        'errors': r'(\d+)\s+errors',
    }
    for key, pat in patterns.items():
        match = re.search(pat, line)
        if match:
            results[key] = int(match.group(1))
    return results


def main():
    parser = argparse.ArgumentParser(description='Parse test results and calculate pass@k metrics')
    parser.add_argument('--path', type=str, 
                        default='/home/jovyan/sukhorukov/repoeval_testing',
                        help='Base directory containing test results')
    parser.add_argument('--vanilla_path', type=str, 
                        default='/home/jovyan/sukhorukov/repoeval_testing',
                        help='Base directory containing test results of vanilla repos')
    parser.add_argument('--prefixes', type=str, nargs='+',
                        default=['ast_cfc'],
                        help='List of prefixes to process')
    parser.add_argument('--k', type=int, default=10,
                        help='K value for pass@k calculation')
    parser.add_argument('--repos', type=str, nargs='+', default=['facebookresearch_omnivore', 'leopard-ai_betty', 'maxhumber_redframes', 'amazon-science_patchcore-inspection', 'deepmind_tracr'],
                        help='List of repositories to process')
    
    
    args = parser.parse_args()
    
    path = args.path
    prefixes = args.prefixes
    k = args.k
    
    vanilla_path = args.vanilla_path
    initially_passed = {
        
    }
    
    for vanilla_res in os.listdir(vanilla_path):
        if vanilla_res.endswith('.out'):
            repo_name = vanilla_res.split('.')[0]
            if repo_name not in args.repos:
                continue
            last_line = get_last_nonempty_line(os.path.join(vanilla_path, vanilla_res))
            result = parse_test_results(last_line)
            initially_passed[repo_name] = result['passed']
    
    # print(f'INITIALLY PASSED: {initially_passed}')
    
    # Collect results
    results = {}
    for prefix in prefixes:
        results[prefix] = {}
        
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist, skipping prefix '{prefix}'")
            continue
            
        for file in sorted(os.listdir(path)):
            if file.endswith('.out'):
                repo_name = '_'.join(file.split('.')[0].split('_')[:-1])
                trial = file.split('.')[0].split('_')[-1]
                if repo_name not in results[prefix]:
                    results[prefix][repo_name] = []
                
                last_line = get_last_nonempty_line(os.path.join(path, file))
                result = parse_test_results(last_line)
                # print(file, last_line, result)
                results[prefix][repo_name].append(result)

    repos = {}
    for key in results[list(results.keys())[0]]:
        repo = '_'.join(key.split('_')[:-1])
        if repo not in repos:
            repos[repo] = 0
        repos[repo] += 1

    metrics = {}
    for prefix in prefixes:
        metrics[prefix] = {}
        for key in results[prefix]:
            repo = '_'.join(key.split('_')[:-1])
            if key not in metrics[prefix]:
                metrics[prefix][key] = []
            for trial in results[prefix][key]:
                # print(trial['passed'], initially_passed[repo])
                if initially_passed[repo] <= trial['passed']:
                    metrics[prefix][key].append(1)
                else:
                    metrics[prefix][key].append(0)


    # Calculate pass@k metrics
    metrics_k = {}
    for prefix in prefixes:
        metrics_k[prefix] = {}
        for key in metrics[prefix]:
            if len(metrics[prefix][key]) < k:
                continue
            repo = '_'.join(key.split('_')[:-1])
            if repo not in metrics_k[prefix]:
                metrics_k[prefix][repo] = []
            
            num_passed = sum(metrics[prefix][key])
            num_total = len(metrics[prefix][key])
            if num_total - num_passed >= k:
                pass_k = 1 - math.comb(num_total - num_passed, k) / math.comb(num_total, k)
            else:
                pass_k = 1.0
            metrics_k[prefix][repo].append(pass_k)
    
    # Calculate final metrics
    final_metrics = {}
    for prefix in prefixes:
        final_metrics[prefix] = {}
        for repo in metrics_k[prefix]:
            final_metrics[prefix][repo] = sum(metrics_k[prefix][repo]) / len(metrics_k[prefix][repo]) * 100
        if prefix == 'ast_cfc':
            print('LLavaCode_AIRI')
        else:
            print(prefix)
        print(f'{final_metrics[prefix]}')
    
    for prefix in prefixes:
        if prefix == 'ast_cfc':
            print('LLavaCode_AIRI')
        else:
            print(prefix)
        total_sum = sum(sum(metrics_k[prefix][repo]) for repo in metrics_k[prefix] if repo in args.repos)
        total_count = sum(len(metrics_k[prefix][repo]) for repo in metrics_k[prefix] if repo in args.repos)
        if total_count > 0:
            print(f'Pass@{k}: {total_sum / total_count * 100:.2f}')
        else:
            print("No data available")


if __name__ == "__main__":
    main()
