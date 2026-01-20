import os
import pickle
import fcntl

def load_vars(filename, catchError=False, is_enumerate=False):
    try:
        value_all = []
        with open(filename,'rb') as f:
            # if is_enumerate:
            #     while True:
            #         value = pickle.load(f)
            #         yield value
            # else:
            try:
                while True:
                    value = pickle.load( f )
                    value_all.append(value)
            except EOFError as e:
                if len(value_all) == 1:
                    return value_all[0]
                else:
                    return value_all
    except Exception as e:
        if catchError:
            warnings.warn( f'Load Error:\n{filename}' )
            return None
        raise e

def save_vars(filename, *vs, verbose=0, append=False):
    if verbose:
        print(f'Save vars to \n{filename}')
    mode = 'ab' if append else 'wb'
    with open(filename, mode) as f:
        if len(vs) == 1:
            pickle.dump(vs[0], f)
        else:
            pickle.dump(vs, f)


def linear_interp(x, x1, y1, x2, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)



class FileLocker:
    '''
    Note! When the variable is released, then the lock will be released.
    '''
    def __init__(self, filename):
        self.__filename = filename
        if not os.path.exists(filename):
            open(filename, "w").close()

    def acquire(self, wait, print_info=False):
        self.file =  open(self.__filename, 'w+')

        if print_info:
            print(f'File Locker acquiring {self.__filename}...', end='')

        if wait:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        else:
            try:
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB )
            except IOError:
                if print_info:
                    print('FAILED')
                return False
        if print_info:
            print('SUCCEED')
        return True

    def release(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()

    def __enter__(self):
        # print(f'acquire file locker {self.__filename}')
        self.acquire(wait=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f'release file locker {self.__filename}')
        self.release()
