import json
from tqdm.auto import tqdm
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Process JSONL file and run bash tests')
    parser.add_argument('--jsonl_file', type=str, required=True,
                        help='Path to the JSONL input file')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Path to the bash script to execute')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to store test results')
    parser.add_argument('--repos', type=str, required=True, nargs='+',
                        default=['amazon-science_patchcore-inspection', 'deepmind_tracr', 'facebookresearch_omnivore', 'leopard-ai_betty', 'maxhumber_redframes'],
                        help='List of repositories to keep')
    
    args = parser.parse_args()
    
    jsonl_file = args.jsonl_file
    # bash_script = args.bash_script
    result_dir = args.result_dir
    
    # os.makedirs('.tmp', exist_ok=True)

    for suffix in ['ast_cfc']:
        for repo_name in args.repos:
            bash_script = os.path.join(args.base_dir, 'repo_tests_function.sh')

            with open(jsonl_file, "r") as fin:
                for i, line in tqdm(enumerate(fin)):
                    
                    entry = json.loads(line)
                    repo = entry["repository"]
                    filepath = entry["filepath"]
                    # filecontent = entry["filecontent"]
                    task_id = entry['task_id']
                    if repo != repo_name:
                        continue
                    for index, filecontent in enumerate(entry['filecontent']):
                        # if os.path.exists(f'/home/jovyan/sukhorukov/LlavaCode/test_results_function_{suffix}_report_final/{repo}_{task_id}_{index}.out'):
                        #     continue

                        with open(f"tmp_filecontent_{repo_name}_function_{suffix}_report_final.txt", "w") as tmp:
                            tmp.write(filecontent)
                            
                        cmd = [
                            bash_script, 
                            repo, 
                            filepath, 
                            f"tmp_filecontent_{repo_name}_function_{suffix}_report_final.txt",
                            task_id + f"_{index}",
                            result_dir,
                            args.base_dir
                        ]

                        result = subprocess.run(cmd)


if __name__ == "__main__":
    main()
