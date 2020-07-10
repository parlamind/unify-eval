import os
import shutil
import sys
import tarfile


class TarWriter:
    """
    Context manager that creates a temporary folder during entering and creates a gzipped tar archive of that folder during exit.
    """

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        # if folder already exists as file (from previous models), overwrite it
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                try:
                    os.remove(self.path)
                except Exception as e:
                    print(e, file=sys.stderr)
                    print(f"could not remove file {self.path}", file=sys.stderr)
            if os.path.isdir(self.path):
                try:
                    shutil.rmtree(self.path)
                except Exception as e:
                    print(e, file=sys.stderr)
                    print(f"could not remove directory {self.path}", file=sys.stderr)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        path_tar = self.path + ".tar"
        with tarfile.open(path_tar, "w:gz") as t:
            for file in os.listdir(self.path):
                full_file_path = os.path.join(self.path, file)
                t.add(full_file_path, arcname=file)
        try:
            shutil.rmtree(self.path)
        except OSError as e:
            print({e}, file=sys.stderr)
            print(f"could not delete uncompressed intermediate model file {self.path}", file=sys.stderr)
            return
        try:
            os.rename(path_tar, self.path)
        except:
            print(f"could not rename compressed intermediate model file {path_tar}", file=sys.stderr)


class TarReader:
    """
    Context manager that temporarily opens a gzipped tar archive as a folder during entering and removes that folder during exit
    """

    def __init__(self, path_to_model_file: str):
        self.path_to_model_file = path_to_model_file
        self._directory = os.path.join(*os.path.split(self.path_to_model_file)[:-1])
        self.path_extracted = os.path.join(self._directory, f"{os.path.split(self.path_to_model_file)[-1]}_extracted")

    def __enter__(self):
        with tarfile.open(self.path_to_model_file, mode="r:gz") as tar:
            try:
                tar.extractall(path=self.path_extracted)
            except Exception as e:
                print(f"exception {e}", file=sys.stderr)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            shutil.rmtree(self.path_extracted)
        except Exception as e:
            print(e, file=sys.stderr)
            print(f"could not remove uncompressed intermediate model file {self.path_extracted}", file=sys.stderr)
