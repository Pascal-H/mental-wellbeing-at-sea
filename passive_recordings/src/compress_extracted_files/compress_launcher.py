import os
import subprocess
import concurrent.futures


def parallel_processing(cur_parent_dir):
    process = subprocess.Popen(
        [
            "python",
            "src/compress_extracted_files/compress_main.py",
            # 'dir-in',
            f"/media/phecker/files/datasets/mental_wellbeing_at_sea/local_deployment/VDR/extracted/raw/{cur_parent_dir}",
            # 'path-interim',
            "/media/phecker/files/datasets/mental_wellbeing_at_sea/local_deployment/VDR/extracted/interim/",
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    while True:
        output = process.stdout.readline()
        print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print("RETURN CODE", return_code)
            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break


def main():
    cur_path_exec = os.path.dirname(os.path.abspath(__file__))
    # Folders to convert to .flac
    lst_extracted = [
        "IMO_9510682_2023-01-30T10-05-31",
        # '',
        # '',
        # '',
        # '',
    ]
    for cur_parent_dir in lst_extracted:
        process = subprocess.Popen(
            [
                "python",
                f"{cur_path_exec}/compress_main.py",
                # 'dir-in',
                f"/media/phecker/files/datasets/mental_wellbeing_at_sea/local_deployment/VDR/extracted/raw/{cur_parent_dir}",
                # 'path-interim',
                "/media/phecker/files/datasets/mental_wellbeing_at_sea/local_deployment/VDR/extracted/interim/",
            ],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        while True:
            output = process.stdout.readline()
            print(output.strip())
            # Do something else
            return_code = process.poll()
            if return_code is not None:
                print("RETURN CODE", return_code)
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip())
                break


if __name__ == "__main__":
    main()
