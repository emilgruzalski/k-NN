import tkinter as tk
import tkinter.filedialog as tkfile
import pandas as pd
from scipy.spatial import distance


class KNNModel:
    def __init__(self):
        self.data = None
        self.norm_data = None
        self.k = 1
        self.metric = "euclidean"
        self.vote = "simple"

    def load_data(self, filename):
        self.data = pd.read_csv(filename, header=None)
        self.norm_data = (self.data - self.data.min()) / (
            self.data.max() - self.data.min()
        )

    def classify(self, point):
        if self.metric == "euclidean":
            dists = self.norm_data.iloc[:, :2].apply(
                lambda row: distance.euclidean(row, point), axis=1
            )
        else:  # manhattan
            dists = self.norm_data.iloc[:, :2].apply(
                lambda row: distance.cityblock(row, point), axis=1
            )

        nearest = dists.nsmallest(self.k)

        if self.vote == "simple":
            votes = self.data.loc[nearest.index, 2].value_counts()
        else:  # weighted
            weights = 1 / nearest**2
            votes = (
                self.data.loc[nearest.index, 2]
                .groupby(self.data.loc[nearest.index, 2])
                .apply(lambda x: (x * weights).sum())
            )

        return int(votes.idxmax()), nearest.index, nearest


class KNNApp:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width=600, height=600)
        self.canvas.pack()
        self.model = KNNModel()
        self.colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        self.last_point = None
        self.neighbors = []
        self.k_slider = tk.Scale(
            master, from_=1, to=20, orient="horizontal", command=self.update_k
        )
        self.k_slider.pack()
        self.metric_menu = tk.OptionMenu(
            master,
            tk.StringVar(value="euclidean"),
            "euclidean",
            "manhattan",
            command=self.update_metric,
        )
        self.metric_menu.pack()
        self.vote_menu = tk.OptionMenu(
            master,
            tk.StringVar(value="simple"),
            "simple",
            "weighted",
            command=self.update_vote,
        )
        self.vote_menu.pack()
        self.load_button = tk.Button(master, text="Load Data", command=self.load_data)
        self.load_button.pack()

    def load_data(self):
        filename = tkfile.askopenfilename()
        if filename:
            self.model.load_data(filename)
            self.draw_points()

    def update_k(self, value):
        self.model.k = int(value)

    def update_metric(self, value):
        self.model.metric = value

    def update_vote(self, value):
        self.model.vote = value

    def draw_points(self):
        self.canvas.delete("all")

        for i in range(len(self.model.norm_data)):
            x = self.model.norm_data.iloc[i, 0] * 550 + 25
            y = self.model.norm_data.iloc[i, 1] * 550 + 25
            color = self.colors[self.model.data.iloc[i, 2]]
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color)

        self.canvas.bind("<Button-1>", self.classify_point)

    def classify_point(self, event):
        x = (event.x - 25) / 550
        y = (event.y - 25) / 550
        category, nearest_indexes, nearest_dists = self.model.classify(
            pd.Series([x, y])
        )
        color = self.colors[category]

        if self.last_point is not None:
            self.canvas.delete(self.last_point)

        self.last_point = self.canvas.create_rectangle(
            event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill=color
        )

        for neighbor in self.neighbors:
            self.canvas.delete(neighbor)

        self.neighbors.clear()

        for i in nearest_indexes:
            nx = self.model.norm_data.iloc[i, 0] * 550 + 25
            ny = self.model.norm_data.iloc[i, 1] * 550 + 25
            self.neighbors.append(
                self.canvas.create_oval(nx - 7, ny - 7, nx + 7, ny + 7, outline="black")
            )
            self.neighbors.append(
                self.canvas.create_text(
                    nx, ny - 10, text=f"{nearest_dists[i]:.2f}", fill="black"
                )
            )


if __name__ == "__main__":
    root = tk.Tk()

    app = KNNApp(root)

    root.mainloop()
