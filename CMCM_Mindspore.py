import cv2
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import context
import mindspore.ops as ops
from mindspore.ops import functional as F
from resnet import resnet50
from dataset import data
from mindspore.common.initializer import initializer, HeNormal
from torch.hub import load_state_dict_from_url


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def conv3x3(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)

def conv1x1(in_channel, out_channel, stride=1, use_se=False):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)

def _fc(in_channel, out_channel, use_se=False):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)

class block(nn.Cell):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

    def construct(self, x):
        while x.size()[2] > 1:    
            x = self._block(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x

class TextCNN(nn.Cell):
    def __init__(self):
        super(TextCNN, self).__init__()
        # conv = nn.Conv2d(1024, 1024, (3, 1), stride=1)
        self.conv = nn.Conv2d(1024, 1024, kernel_size=(3,1), stride=1, padding=0, pad_mode='same')
        self.func1 = nn.SequentialCell(
            # nn.Conv2d(1, 1024, (3, 300), stride=1),
            nn.Conv2d(1, 1024, kernel_size=(3,300), stride=1, padding=0, pad_mode='same'),
            # nn.ZeroPad2d((0, 0, 1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(3,1), stride=1, padding=0, pad_mode='same'),
            # nn.ZeroPad2d((0, 0, 1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(3,1), stride=1, padding=0, pad_mode='same'),
        )
        self.relu = nn.ReLU()
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.fc = nn.Dense(1024, 200)
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        x = self.func1(x)
        x = self.padding2(x)
        px = self.max_pool(x)
        # x = self.padding1(px)        
        x = self.relu(px)
        x = self.conv(x)

        # x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.relu(px)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x + px
        
        x = x.squeeze()
        return x

class resnet_video(nn.Cell): 
    def __init__(self, pretrained=False):
        super().__init__()
        image_model = resnet50(class_num=200)
        self.cnn_image = nn.SequentialCell(*list(image_model.cells())[:-1])
        self.fc=list(image_model.cells())[-1]

    def construct(self, x, is_cls=True):
        x = self.cnn_image(x).squeeze()
        if is_cls:
            x=self.fc(x)
        return x

def calculate_r(pred,a,b,c,d):
    return pred

class CMNet(nn.Cell):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_img = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        self.conv1_aud = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        self.conv1_vid = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        # self.conv1_vid = nn.Conv2d(in_channels=2048, out_channels=682, kernel_size=1)
        self.conv1_test = nn.Conv2d(in_channels=1024*3, out_channels=1024, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #image
        image_model = resnet50(class_num=200)

        self.cnn_image = nn.SequentialCell(*list(image_model.cells())[:-2])
        #audio
        audio_model = resnet50(class_num=200)
        self.cnn_audio = nn.SequentialCell(*list(audio_model.cells())[:-2])
        #text
        text_model = TextCNN()
        self.cnn_text = text_model#nn.SequentialCell(*list(text_model.cells())[:-1])
        #video
        video_model = resnet_video()
        self.cnn_video = nn.SequentialCell(*list(list(video_model.cells())[:-1][0])[:-1])

        self.last_layer1 = nn.AdaptiveAvgPool2d(1)
        self.last_layer2 = nn.Dense(1024*4, 200)
        self.fc = nn.Dense(2048, 200)
        
        self.concat_op = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        
    def construct(self, x1,x2,x3,x4): #image,audio,video,text
        x1 = self.cnn_image(x1)#.squeeze()  #bs*2048
        x2 = self.cnn_audio(x2)#.squeeze()
        x3 = self.cnn_video(x3)#2048*14*14
        x1 = self.conv1_img(x1)
        x2 = self.conv1_aud(x2)
        x3 = self.conv1_vid(x3)#1024*14*14
        x1_1 = self.avgpool(x1).squeeze()
        x2_2 = self.avgpool(x2).squeeze()
        x3_3 = self.avgpool(x3).squeeze()#1024*1*1

        # x4 = x4.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        # x4 = self.expand_dims(x4, 3)
        x4 = self.expand_dims(x4, 1)
        # print(x4.shape)
        x4 = self.cnn_text(x4)
        x4_4 = x4.squeeze()  # [batch_size, num_filters(250)]

        x = self.concat_op((x1, x2, x3))
        x = self.last_layer1(x)
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        # print(x.shape)
        
        x4 = self.conv1_test(x4.reshape(x4.shape[0],x4.shape[1]*x4.shape[2],1,1))
        x = self.concat_op((x, x4))
        x = x.squeeze()
        x = self.last_layer2(x)

        return x, x1_1, x2_2, x3_3, x4_4


class CenterLoss(nn.Cell):
    """
    paper: http://ydwen.github.io/papers/WenECCV16.pdf
    code:  https://github.com/pangyupo/mxnet_center_loss
    pytorch code: https://blog.csdn.net/sinat_37787331/article/details/80296964
    """
    def __init__(self, features_dim, num_class=200, lamda=0.01, scale=1., batch_size=64):
        super(CenterLoss, self).__init__()
        self.lamda = lamda
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        # store the center of each class , should be ( num_class, features_dim)
        self.feature_centers = ms.Parameter(Tensor(np.random.randn([num_class, features_dim]), ms.float32), name="feature_centers", requires_grad=True)
        # self.lossfunc = CenterLossFunc.apply

    def construct(self, output_features, y_truth):
        batch_size = y_truth.size(0)
        output_features = output_features.view(batch_size, -1)
        # print(output_features.size(-1))
        assert output_features.size(-1) == self.feat_dim
        factor = self.scale / batch_size
        # return self.lamda * factor * self.lossfunc(output_features, y_truth, self.feature_centers))

        # print(self.feature_centers.shape)
        tmp_y_truth = y_truth.unique()
        diff_centers = self.feature_centers.index_select(0, tmp_y_truth.long())
        # print(diff_centers.shape)
        center_num = diff_centers.size(0)
        diff_centers_1 = diff_centers.view(center_num, 1, 1024)
        diff_centers_2 = diff_centers.view(1, center_num, 1024)
        # dist = scipy.spatial.distance.cdist(diff_centers_1, diff_centers_2, 'cosine')
        loss2 = 10 - 0.1*ms.cosine_similarity(diff_centers_1, diff_centers_2, dim=1).sum()/2/center_num
        # center_num = self.feature_centers.size(0)
        # diff_centers_1 = self.feature_centers.view(center_num, 1, 1024)
        # diff_centers_2 = self.feature_centers.view(1, center_num, 1024)
        # loss2 = 410-self.lamda*(diff_centers_1 - diff_centers_2).pow(2).sum()/2/1024

        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_batch
        loss = self.lamda * 0.5 * factor * (diff.pow(2).sum())
        #########
        return loss+loss2


if __name__ == '__main__':
    device = 'GPU'
    batch_size=2
    num_worker=4
    retrival=0
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
    # context.set_context(mode=context.GRAPH_MODE, device_target=device)
    network = CMNet()
    train_dataloader = data(train=True, batch_size=batch_size, num_worker=num_worker)
    test_dataloader = data(train=False, batch_size=4, num_worker=4)

    optimizer = nn.SGD(params=network.trainable_params(), learning_rate=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()


    def forward_fn(image,audio,text,video, targets):
        logits,a,b,c,d = network(image,audio,video,text)
        loss = loss_fn(logits, targets)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(image,audio,text,video, targets):
        loss, grads = grad_fn(image,audio,text,video, targets)
        optimizer(grads)
        return loss

    for i in range(50):

        network.set_train()
        for idx, (image, audio, text, video, class_id) in enumerate(train_dataloader):
            class_id=class_id.int()
            loss = train_step(image,audio,text,video, class_id)
            print(loss)
            if idx % (100/batch_size) == 0:
                loss, current = loss.asnumpy(), idx
                print(f"loss: {loss:>7f}  [{current:>3d}/{5994:>3d}]")
        correct=0
        for idx, (image, audio, text, video, class_id) in enumerate(test_dataloader):
            class_id = class_id.int()
            pred,a,b,c,d = network(image,audio,video,text)
            correct += (pred.argmax(1) == class_id).asnumpy().sum()
            print(correct)
            if (retrival==True):
                it,ia,iv,ti,ta,tv,ai,at,av,vi,vt,va,a=calculate_r(pred,a,b,c,d)
                print(correct)

                print('Epoch = ', i)
                print('__________________________')
                print('I--T =', it, 'I--A =', ia, 'I--V =', iv, )
                print('T--I =', ti, 'T--A =', ta, 'T--V =', tv, )
                print('A--I =', ai, 'A--T =', at, 'A--V =', av, )
                print('V--I =', vi, 'V--T =', vt, 'V--A =', va, )
                print('Average =', a)







