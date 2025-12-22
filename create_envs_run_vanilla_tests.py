# import subprocess

# for repo in ['amazon-science_patchcore-inspection', 'deepmind_tracr', 'facebookresearch_omnivore', 'leopard-ai_betty', 'maxhumber_redframes']:
#     subprocess.run(['bash', 'testing_functions.sh', repo])
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run bash script for multiple repositories')
    parser.add_argument('--base_dir', type=str,
                        help='Base directory path')
    parser.add_argument('--python_version', type=str, default='3.10',
                        help='Python version for conda environment')
    parser.add_argument('--bash_script', type=str, default='testing_functions.sh',
                        help='Path to the bash script')
    parser.add_argument('--results_folder_name', type=str, default='results_vanilla_functions_report_final',
                        help='Name of the folder to store results')
    parser.add_argument('--repos', type=str, nargs='+',
                        default=['amazon-science_patchcore-inspection', 'deepmind_tracr', 
                                'facebookresearch_omnivore', 'leopard-ai_betty', 'maxhumber_redframes'],
                        help='List of repositories to process')
    
    args = parser.parse_args()
    
    for repo in args.repos:
        subprocess.run([
            'bash', args.bash_script,
            '--repo_path', repo,
            '--base_dir', args.base_dir,
            '--python_version', args.python_version,
            '--results_folder_name', args.results_folder_name,
        ])

if __name__ == "__main__":
    main()
