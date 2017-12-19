from slacker import Slacker
from .config import *

slack = Slacker(slack_api)

def send_notification(msg):
    print('sending message to slack ...')
    slack.chat.post_message('#general', msg)
    print('message sent')

def compose_iteration_msg(iter_cnt, total_iter_cnt, test_score):
    msg1 = 'iteration ' + str(iter_cnt) + ' out of ' + str(total_iter_cnt) + '\n'
    msg2 = 'testing loss: ' + str(test_score[0]) + '\n'
    msg3 = 'testing accuracy: ' + str(test_score[1]) + '\n'
    msg = msg1 + msg2 + msg3
    return msg
