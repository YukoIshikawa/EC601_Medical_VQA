import json
import urllib.request
import os
import re
import numpy as np
from tqdm import trange
from collections import defaultdict
from PIL import Image
from sklearn import preprocessing
import text_helper
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from medical_models import VqaModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = {transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])}

SAVE_PATH = './img'
RESIZE_PATH = './Resize_img'
DATA_PATH = './Data'
DATASET = './VQA_RAD Dataset Public.json'
PIC_SIZE = [224, 224]

# save pictures from json
def save_pic():
    j = open(DATASET)
    info = json.load(j)
    unable_link = []
    for i in trange(len(info)):
        tag = download_image(info[i]['image_name'],info[i]['image_case_url'])
        if tag == 0:
            unable_link.append(i)
    print("\nUnable links: ",unable_link) # [584, 992, 1351, 1376, 1394, 1406, 1407, 1414, 1415, 1725, 1972]

# download by url
def download_image(name, url):
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    request = urllib.request.Request(url)
    request.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36')
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        # print("status code", e.code)
        # print("reson", e.reason)
        # print("header", e.headers)
        return 0
    buf = response.read()
    buf = str(buf, encoding='utf-8')
    listurl = re.findall(r'http.+\.jpg', buf)

    req = urllib.request.urlopen(listurl[0])
    buf = req.read()
    if not os.path.exists(SAVE_PATH + '/' + name):
        with open(SAVE_PATH + '/' + name, 'wb') as f:
            f.write(buf)
    return 1

# look for the longest question
def question_length():
    j = open(DATASET)
    info = json.load(j)
    longest_q = 0
    longest_q_id = 0
    for information in info:
        l = len(text_helper.tokenize(information['question']))
        if longest_q < l:
            longest_q = l
            longest_q_id = information['qid']
    question = text_helper.tokenize(info[longest_q_id]['question'])
    print(longest_q, longest_q_id, question)

# make dictionary for question tokens
def make_question_vocabulary():
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    j = open(DATASET)
    info = json.load(j)
    set_question_length = [None]*len(info)
    for i, infomation in enumerate(info):
        words = SENTENCE_SPLIT_REGEX.split(infomation['question'].lower())
        words = [w.strip() for w in words if len(w.strip()) > 0]
        vocab_set.update(words)
        set_question_length[i] = len(words)
    question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    
    with open(DATA_PATH+'/medical_questions_vocab.txt', 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])
    
    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))

# make dictionary for answers
def make_answer_vocabulary():
    answers = defaultdict(lambda: 0)
    j = open(DATASET)
    info = json.load(j)
    for information in info:
        word = information['answer']
        if type(word) == type('a'):
            word = word.lower()
        answers[word] += 1
                
    answers = sorted(answers, key=answers.get, reverse=True)
    assert('<unk>' not in answers)
    answers = ['<unk>'] + answers
    
    with open(DATA_PATH+'/medical_answers_vocab.txt', 'w') as f:
        f.writelines([str(w)+'\n' for w in answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))

# resize the picture to [224, 224]
def resize_images():
    if not os.path.exists(RESIZE_PATH):
        os.mkdir(RESIZE_PATH)
    images = os.listdir(SAVE_PATH)
    failed_images = []

    for iimage, image in enumerate(images):
        try:
            with open(os.path.join(SAVE_PATH + '/', image), 'r+b') as f:
                with Image.open(f) as img:
                    # img = img.resize([min_edge,min_edge], Image.ANTIALIAS)
                    img = img.resize(PIC_SIZE, Image.ANTIALIAS)
                    img.save(os.path.join(RESIZE_PATH + '/', image), img.format)
        except(IOError, SyntaxError) as e:
            failed_images.append(iimage)

# build inputs
def build_medical_input():
    j = open(DATASET)
    info = json.load(j)
    dataset = [None]*len(info)
    for n_info, information in enumerate(info):
        image_name=information['image_name']
        image_path=os.path.join(RESIZE_PATH + '/', image_name)
        question_id=information['qid']
        question_str=information['question']
        question_tokens=text_helper.tokenize(question_str)
        all_answers = [str(information['answer'])]
        iminfo = dict(
            image_name = image_name,
            image_path = image_path,
            question_id = question_id,
            question_str = question_str,
            question_tokens = question_tokens,
            all_answers = all_answers,
            valid_answers = all_answers)

        dataset[n_info] = iminfo
    data_array = np.array(dataset)
    # np.savetxt("./med_input.txt",data_array, fmt='%s')
    length = len(data_array)
    train_arr = data_array
    valid_arr = data_array[int(length*0.9)+1:length]

    np.save(DATA_PATH+'/train.npy', train_arr)
    np.save(DATA_PATH+'/valid.npy', valid_arr)
    np.save(DATA_PATH+'/train_valid.npy', data_array)

def build_dataset(DATASET):
    j = open(DATASET)
    info = json.load(j)
    
    count = 0
    output = []
    for data in info:
        data['question'] = data['question'].replace('/Are', '').replace('/are', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '').replace(',', '') 
        if "? -yes/no" in data['question']:
            data['question'] = data['question'].replace("? -yes/no", "")
        if "? -open" in data['question']:
            data['question'] = data['question'].replace("? -open", "")
        if "? - open" in data['question']:
            data['question'] = data['question'].replace("? - open", "")
        d = { 
            'image_name': data['image_name'],
            'image_path': './Resize_img/{}'.format(data['image_name']),
            'question_id': count,
            'question_str': data['question'],
            'question_tokens': text_helper.tokenize(data['question']),
            'all_answers': [data['answer']], 
            'valid_answers': [data['answer']]
        }
        output.append(d)
        count += 1
    if (DATASET == "trainset.json"):    
        with open('train.npy', 'wb') as f:
            np.save(f, output)
    elif (DATASET == "testset.json"):
        with open('valid.npy', 'wb') as f:
            np.save(f, output)

###############################################

# make dictionary for question tokens
def make_question_vocab(DATASET):
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    j = open(DATASET)
    info = json.load(j)
    set_question_length = [None]*len(info)
    for i, infomation in enumerate(info):
        words = SENTENCE_SPLIT_REGEX.split(infomation['question'].lower())
        words = [w.strip() for w in words if len(w.strip()) > 0]
        vocab_set.update(words)
        set_question_length[i] = len(words)
    question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    
    with open(DATA_PATH+'/medical_questions_vocab.txt', 'w', encoding='utf-8') as f:
        f.writelines([w+'\n' for w in vocab_list])
    
    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))

# make dictionary for answers
def make_answer_vocab(DATASET):
    answers = defaultdict(lambda: 0)
    j = open(DATASET)
    info = json.load(j)
    for information in info:
        word = information['answer']
        if type(word) == type('a'):
            word = word.lower()
        answers[word] += 1
                
    answers = sorted(answers, key=answers.get, reverse=True)
    assert('<unk>' not in answers)
    answers = ['<unk>'] + answers
    
    with open(DATA_PATH+'/medical_answers_vocab.txt', 'w', encoding='utf-8') as f:
        f.writelines([str(w)+'\n' for w in answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))

###############################################

# calculate normalization of these medical images
# mean = [0.2539457, 0.2538943, 0.25384054]
# std= [0.048477963, 0.048473958, 0.04845482]
def look_for_normalization():
    images = os.listdir(RESIZE_PATH)
    images_list = []
    for i, image in enumerate(images):
        img = Image.open(RESIZE_PATH + '/' +image).convert('RGB')
        transf = transforms.ToTensor()
        images_list.append(transf(img).numpy())
    images_array = np.array(images_list)
    images_array.resize(len(images),3,224*224)
    img_mean = np.mean(images_array,axis=0)
    img_std = np.std(images_array,axis=0)
    mean = [0,0,0]
    std = [0,0,0]
    for i in range(3):
        mean[i] = np.mean(img_mean[i],axis=0)
        std[i] = np.std(img_std[i],axis=0)
    print(mean, std, sep='\n')

def predict(image_name, question):

    img = Image.open('./Resize_img/' + image_name)
    image = img.resize(PIC_SIZE, Image.ANTIALIAS)
    image = image.convert('RGB')
    for trans in transform:
        image = trans(image)
    image = image.view(1,3,224,224)
    
    qst_vocab = text_helper.VocabDict('./Data/medical_questions_vocab.txt')
    ans_vocab = text_helper.VocabDict('./Data/medical_answers_vocab.txt')
    qst = question.lower().split()
    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * 30)
    qst2idc[:len(qst)] = [qst_vocab.word2idx(w) for w in qst]
    qst2idc = qst2idc.reshape(1,30)

    checkpoint = torch.load('./models/model-epoch-10.ckpt', map_location=torch.device('cpu'))
    model = VqaModel(
        embed_size=1024,
        qst_vocab_size=qst_vocab.vocab_size,
        ans_vocab_size=ans_vocab.vocab_size,
        word_embed_size=300,
        num_layers=2,
        hidden_size=512)
    model.load_state_dict(checkpoint['state_dict'])
    model_ = model.to(device)
    model_.eval()
    torch.no_grad()
    img_ = image.to(device)
    qst_ = torch.from_numpy(qst2idc).to(device)

    output = model_(img_,qst_)

    v, pred = torch.sort(output,1,True)
    index_arr = pred.cpu().detach().numpy()
    value_arr = v.cpu().detach().numpy()
    print('image:'.ljust(15),image_name)
    print('question:'.ljust(15), question)
    print('answers'.ljust(40),'possibility')
    for i in range(5):
        print(ans_vocab.idx2word(index_arr[0][i]).ljust(40),value_arr[0][i])
    print('\n')

class inf():
    def __init__(self,phase, epoch,loss, acc1, acc2):
        self.phase = phase
        self.epoch = epoch
        self.loss = loss
        self.acc_exp1 = acc1
        self.acc_exp2 = acc2

    def __str__(self):
        return self.phase+' '+self.epoch+' '+self.loss+' '+self.acc_exp1+' '+self.acc_exp2

def check_output():
    inf_list = []
    epoch = 10
    for i in range(epoch):
        for phase in ['train', 'valid']:
            with open(os.path.join('.\logs', '{}-log-epoch-{:02}.txt').format(phase, i+1), 'r') as f:
                line = f.readline()
                words = line.split()
                information = inf(phase,words[0],words[1],words[2],words[3])
                inf_list.append(information)
    loss_list_t = []
    loss_list_v = []
    acc1_list_t = []
    acc1_list_v = []
    acc2_list_t = []
    acc2_list_v = []
    for i in range(epoch * 2):
        if inf_list[i].phase=='train':
            loss_list_t.append(round(float(inf_list[i].loss),4))
            acc1_list_t.append(round(float(inf_list[i].acc_exp1),4))
            acc2_list_t.append(round(float(inf_list[i].acc_exp2),4))
        if inf_list[i].phase=='valid':
            loss_list_v.append(round(float(inf_list[i].loss),4))
            acc1_list_v.append(round(float(inf_list[i].acc_exp1),4))
            acc2_list_v.append(round(float(inf_list[i].acc_exp2),4))
    
    fig = plt.figure(1)
    ax1 = plt.subplot(2,2,1)
    plt.plot(range(epoch),loss_list_t,label='train')
    plt.plot(range(epoch),loss_list_v,label='valid')
    ax2 = plt.subplot(2,2,3)
    plt.plot(range(epoch),acc1_list_t)
    plt.plot(range(epoch),acc1_list_v)
    ax3 = plt.subplot(2,2,4)
    plt.plot(range(epoch),acc2_list_t)
    plt.plot(range(epoch),acc2_list_v)
    fig.legend()
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    # save_pic()
    #make_question_vocab("trainset.json")
    #make_answer_vocab("trainset.json")
    # question_length()
    # resize_images()
    # build_medical_input()
    # look_for_normalization()
    check_output()


    # qid: 17
    predict('synpic27142.jpg','What organ system is pictured ?')
    predict('synpic19114.jpg','What organ system is pictured ?')
    # qid: 24
    predict('synpic27142.jpg','Is brain pictured ?')
    predict('synpic27142.jpg','Is chest pictured ?')

    predict('synpic27142.jpg','Is the heart enlarged ?')
    predict('synpic27142.jpg','Is this an MRI ?')
    predict('synpic19114.jpg','Is this an MRI ?')
    # qid: 16
    predict('synpic27142.jpg','What type of imaging is this ?')
    predict('synpic19114.jpg','What type of imaging is this ?')

    #build_dataset("trainset.json")
    #build_dataset("testset.json")
