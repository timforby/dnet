import errno
import os


def write_for_latex(file, epoch, question, value: float):
    """
    Write values in a text file. Data are used in latex graphs.
    :param file: String name of the file
    :param epoch: Integer value of the epoch number
    :param value: Value corresponding to the epoch
    """
    path = file + "_" + question
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, "a") as f:
        text = str(epoch) + " " + str(value) + "\n"
        f.write(text)


def store_in_text(path, filename, to_print):
    f = open(path + '/' + filename + '.txt', 'w')
    f.write(str(to_print))
    f.close()
