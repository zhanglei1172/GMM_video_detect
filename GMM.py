import numpy as np
# import cv2
import glob #, os
# import matplotlib.image as mpig
import matplotlib.pyplot as plt
import matplotlib.animation as animation

EPSILON = 1e-7
DIM = 3  # RGB


class GMMs():
    def __init__(self, components=4, alpha=0.01):
        self.components = components
        self.alpha = alpha

    def train(self, file_list):
        assert file_list
        img_shape = plt.imread(file_list[0]).shape
        self.row_num, self.col_num = img_shape[0], img_shape[1]
        self.weights = np.random.randn(*(self.row_num, self.col_num,
                                         self.components))
        self.weights /= self.weights.sum(axis=-1, keepdims=True)
        self.mus = np.ones((self.components, DIM)) / 2.
        self.sigmas = np.tile(
            np.eye(DIM) / 2.,
            [self.row_num, self.col_num, self.components, 1, 1])
        # self.gmm = [[GMM(self.components) for j in range(self.col_num)]
        #             for i in range(self.row_num)]
        for it, file in enumerate(file_list):
            print('training frame: {}'.format(it))
            img = plt.imread(file) / 255.
            self.update(img)

    def test(self, file_list):
        assert file_list
        res = []
        res_cat = []
        for file in file_list:
            img = plt.imread(file) / 255.
            res.append((self.test_frame(img) * 255).astype('uint8'))
            res_cat.append(
                np.concatenate([(img * 255).astype('uint8'), res[-1]], axis=1))
        return res, res_cat

    def test_frame(self, frame):
        result = frame.copy()
        mask = self.bg_mask(frame)
        result[(mask & (self.weights >= 1 / self.components)).any(-1)] = 0
        return result

    def bg_mask(self, x):
        x_ = np.expand_dims(x, -2) - self.mus
        d = np.squeeze(
            np.sqrt(
                np.expand_dims(x_, -2) @ np.linalg.inv(self.sigmas)
                @ np.expand_dims(x_, -1)))
        return d < 2.5

    def update(self, x):
        mask = self.bg_mask(x)
        self.weights = self.weights + self.alpha * (mask - self.weights)
        x_ = np.expand_dims(x, -2) - self.mus
        self.mus = self.mus + np.expand_dims(
            mask * (self.alpha / (self.weights + EPSILON)), -1) * x_
        self.sigmas = self.sigmas + np.expand_dims(mask*self.alpha / (self.weights + EPSILON), (-1,-2))\
                                    * (np.matmul(np.expand_dims(x_,-1), np.expand_dims(x_,-2)) - self.sigmas)
        mask_reinit = (~mask).all(axis=-1)
        idxs = np.argmin(self.weights[mask_reinit], axis=-1)
        self.mus[mask_reinit][:, idxs, ...] = np.tile(
            x[mask_reinit].reshape(-1, 1, DIM), [1, len(idxs), 1])
        self.sigmas[mask_reinit][:, idxs, ...] = np.tile(
            np.eye(DIM).reshape(-1, 1, DIM, DIM),
            [len(self.mus[mask_reinit]),
             len(idxs), 1, 1])

        # normalize weight
        self.weights /= self.weights.sum(axis=-1, keepdims=True)

    # def save_to_vedio(self, res_cat):
    #     if not os.path.exists('./out'):
    #         os.mkdir('./out')
    #     writer = cv2.VideoWriter('./out/out.mp4', cv2.VideoWriter_fourcc('m','p','4','v'),10, (self.col_num*2, self.row_num), True)
    #     for frame in res_cat:
    #         writer.write(frame)

    #     writer.release()


# def write_to_file(filename,frame, pdir='./out'):
#     if not os.path.exists(pdir):
#         os.mkdir(pdir)
#     cv2.imwrite('{}/{}'.format(pdir,filename), frame)



def gui(res_cat):
    snapshots = res_cat
    nSeconds = 5
    fps = len(res_cat) // nSeconds
    
    

    fig = plt.figure( figsize=(15,8) )

    a = snapshots[0]
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = len(res_cat),
                                interval = 1000 / fps, # in ms
                                )
    plt.show()
    # anim.save('out.gif', fps=fps, writer='imagemagick')

    print('Done!')

if __name__ == '__main__':
    np.random.seed(0)
    data_path = input('请输入WavingTrees所在路径，默认值为"./WavingTrees"') or './WavingTrees' #'/home/zhang/Downloads/Compressed/WavingTrees'
    all_frame = sorted(glob.glob(r'{}/b*.bmp'.format(data_path)))

    model = GMMs()
    model.train(all_frame[:200])
    print('begin test...')
    result, res_cat = model.test(all_frame[200:])
    # model.save_to_vedio(res_cat)
    # cap = cv2.VideoCapture("./out/out.mp4")
    # while(1):
    #     # get a frame
    #     ret, frame = cap.read()
    #     # show a frame
    #     if not ret:
    #         break
    #     cv2.imshow("capture", frame)
    #     if cv2.waitKey(100) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # import matplotlib.pyplot as plt
    # plt.imshow(result)
    # plt.show()
    gui(res_cat)
    print('finished!')
    
