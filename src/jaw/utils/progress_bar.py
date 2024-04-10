"""
Simple progress bar. All credits to Kangliu : https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py.
"""

import os
import sys
import time


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current: int, total: int, msg: str = None):
    """
    Print a progress bar.

    .. note::

       Call this function inside the :func: `test` function. Example :

       .. code-block:: python

        def test(model, loader, f_loss, optimizer, device):
            
            ...

            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = f_loss(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                N += inputs.shape[0]
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
                predicted_targets = outputs.argmax(dim=1)
                correct += (predicted_targets == targets).sum().item()

                progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (tot_loss/(i+1), 100.*correct/N, correct, N))

            return tot_loss/N, correct/N

    :param current: The position of the current example inside the dataset.
    :type current: int.
    :param total: Number of total examples inside the dataset.
    :type total: int.
    :param msg: Message to print aside the progress bar.
    :type msg: str.

    :returns: the string of the formated elapsed time.
    """

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds: int):
    """
    Transform second into days, hours or minutes if needed. Format the time as follow : day, hours, min, sec, ms.

    :param seconds: Seconds elapsed since the training launching.
    :type seconds: int.

    :returns: the string of the formated elapsed time.
    """

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'

    return f