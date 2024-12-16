import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from sklearn.preprocessing import StandardScaler

class FCM:
    def __init__(self, n_clusters=3, max_iter=300, m=2, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state
        self.cluster_centers_ = None
        self.u = None
        self.labels_ = None

    def fit(self, X, init_centers=None, normalize=False):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 是否归一化
        if normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = None

        # 是否提供有初始聚类中心
        if init_centers is not None:
            self.cluster_centers_ = init_centers
        else:
            random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.cluster_centers_ = X[random_idx]

        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=X.shape[0])

        for _ in range(self.max_iter):
            u_old = np.copy(self.u)

            # 更新聚类中心
            self.cluster_centers_ = self._update_centroids(X)

            # 更新隶属度矩阵U
            self.u = self._update_membership(X)

            # 检查收敛性
            if np.linalg.norm(self.u - u_old) < self.error:
                break

        # 获取聚类标签
        self.labels_ = np.argmax(self.u, axis=1)

    def _update_centroids(self, X):
        um = self.u ** self.m
        centroids = np.dot(um.T, X) / np.sum(um.T, axis=1)[:, np.newaxis]
        return centroids

    def _update_membership(self, X):
        power = 2 / (self.m - 1)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        u = 1.0 / (distances ** power)
        u = u / np.sum(u, axis=1)[:, np.newaxis]
        return u

    def predict(self, X, normalize=False):
        if self.scaler is not None and normalize:
            X = self.scaler.transform(X)

        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        u = 1.0 / (distances ** (2 / (self.m - 1)))
        u = u / np.sum(u, axis=1)[:, np.newaxis]
        return np.argmax(u, axis=1)

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, init_centers=None, normalize=False):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 是否归一化
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # 是否提供有初始聚类中心
        if init_centers is not None:
            self.cluster_centers_ = init_centers
        else:
            random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.cluster_centers_ = X[random_idx]

        for _ in range(self.max_iter):
            # 计算每个点到聚类中心的距离，并分配到最近的聚类中心
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # 更新聚类中心
            new_centers = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])

            if np.all(self.cluster_centers_ == new_centers):
                break
            self.cluster_centers_ = new_centers

    def predict(self, X, normalize=False):
        # 如果需要归一化数据
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        distance = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distance, axis=1)

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation with KMeans and FCM")
        
        self.k_value = tk.IntVar(value=3)
        
        # 创建选择图片按钮
        self.select_button = tk.Button(root, text="Select Image", command=self.load_image)
        self.select_button.pack()
        
        # 创建k值调整滑块
        self.k_slider = tk.Scale(root, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Clusters (k)", variable=self.k_value, command=self.update_segmentation)
        self.k_slider.pack()
        
        # 创建画布用于显示图像
        self.figure, self.axs = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()
        
        self.image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.convert('RGB')  # 确保图像是 RGB 格式
            img = img.resize((128, 128))  # 调整图像大小
            self.image = np.array(img) / 255.0  # 归一化
            self.update_segmentation()

    def update_segmentation(self, event=None):
        if self.image is not None:
            img_reshaped = self.image.reshape(-1, 3)
            
            # KMeans 分割
            kmeans = KMeans(n_clusters=self.k_value.get(), random_state=42)
            kmeans.fit(img_reshaped, normalize=True)
            kmeans_labels = kmeans.predict(img_reshaped, normalize=True)
            kmeans_segmented = kmeans_labels.reshape(self.image.shape[:2])
            
            # FCM 分割
            fcm = FCM(n_clusters=self.k_value.get(), random_state=42)
            fcm.fit(img_reshaped, normalize=True)
            fcm_labels = fcm.predict(img_reshaped, normalize=True)
            fcm_segmented = fcm_labels.reshape(self.image.shape[:2])
            
            # 显示原图
            self.axs[0].imshow(self.image)
            self.axs[0].set_title("Original Image")
            self.axs[0].axis('off')
            
            # 显示 KMeans 分割结果
            self.axs[1].imshow(kmeans_segmented, cmap='viridis')
            self.axs[1].set_title("KMeans Segmentation")
            self.axs[1].axis('off')
            
            # 显示 FCM 分割结果
            self.axs[2].imshow(fcm_segmented, cmap='viridis')
            self.axs[2].set_title("FCM Segmentation")
            self.axs[2].axis('off')
            
            self.canvas.draw()

# 创建主窗口
root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()