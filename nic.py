import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from scipy import linalg
import math
import random


# settings
np.random.seed(3487237)
random.seed(a=3487237)  # this was not set when the plots were exported


# data variables
n = 500
x1_mu = [30, 0]
x1_deviation = [1, 4]
x2_mu = [0, 0]
x2_deviation = [1, 1]
x3_radius = 15
x3_deviation = 3
n_clusters = 3
n_iterations = 50


# run variables
calculate = True
plot = True
save_plots = False


# plot stuff
colors = ["#1f3ec4", "#c41f56", "#009900"]
if plot:
    color = []
    for i in range(0, n_clusters):
        color += n * [colors[i]]
    color = np.array(color, dtype=object)

def plot(points, color, title='', block=True):
    for i in range(0, len(color)):
        r, g, b = to_rgb(color[i])
        color[i] = (r, g, b, 0.5)
    plt.clf()
    plt.axis('equal')
    plt.title(title)
    plt.scatter(points[:,0], points[:,1], c=color, s=8)
    # plt.draw()
    plt.pause(0.1)
    plt_block = not block or save_plots
    plt.show(block=not plt_block)
    if save_plots and block:
        title = title.split(',')[0].lower().replace(' ', '').replace(',', '').replace(':', '')
        plt.savefig('/home/daniel/Downloads/{}'.format(title))


# calculate the score of the assignment
def get_score(x, y):
    assert len(x) == len(y)
    score = 0
    for j in range(0, n_clusters):
        inner_sum = 0
        # calculate sum_2 only with the points of the same cluster
        relevant_points = x[y==j]
        # remove e
        for point1 in relevant_points:
            # remove point1 from relevant points, because the distance would be 0 and log(0) doesn't work
            relevant_points_without_point1 = relevant_points[(relevant_points != point1).T[0]]
            # calculate the distances
            distances = np.linalg.norm(relevant_points_without_point1 - point1, axis=1)
            # take the log of the distances
            log_distances = np.log(distances)
            # sum up the log distances
            inner_sum += np.sum(log_distances)
        # add the score of the cluster to the overall score
        score += (1 / (len(relevant_points) - 1)) * inner_sum
    # return the score
    return round(score, 2)


# data
x1 = np.random.normal(x1_mu, x1_deviation, (n, 2))
phi = - math.pi / 4
x1 = x1.dot(np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]]))

x2 = np.random.normal(x2_mu, x2_deviation, (n, 2))

x3_1 = []
x3_2 = []
for i in range(0, n):
    angle = random.uniform(0,1)*(math.pi*2)
    x3_1.append(x3_radius * math.cos(angle) + random.uniform(-x3_deviation, x3_deviation))
    x3_2.append(x3_radius * math.sin(angle) + random.uniform(-x3_deviation, x3_deviation))
x3 = np.array([x3_1, x3_2]).T

y = []
for i in range(0, n_clusters):
    y += n * [i]
y = np.array(y)

# concat
x = np.array([]).reshape(0, 2)
for i in range(0, n_clusters):
    k = i + 1
    next_x = globals()['x{}'.format(k)]
    x = np.concatenate((x, next_x))
    assert np.allclose(x[i*n:k*n], next_x)
if plot:
    plot(x, color, title='Raw Data, Score: {}'.format(get_score(x, y)))


# calculate matrix to whiten the data
cov = np.cov(x.T)
cov_inv = np.linalg.inv(cov)
cov_inv_sqroot = linalg.sqrtm(cov_inv)
x_whitened = x.dot(cov_inv_sqroot)
if plot:
    plot(x_whitened, color, title='Whitened Data, Score: {}'.format(get_score(x_whitened, y)))


# random assignment of the points to clusters
x = x_whitened
np.random.shuffle(x)
y = np.random.randint(n_clusters, size=len(x))
if plot:
    for i in range(0, n_clusters):
        color[y==i] = colors[i]
    plot(x, color, title='Random Assignment, Score: {}'.format(get_score(x, y)))


# improve the score
def improve_score(x, y):
    score = get_score(x, y)
    for k in range(1, n_iterations + 1):
        # iterate over allpoints
        for i in range(0, len(y)):
            for j in range(0, n_clusters):
                # skip if nothing changed
                if j == y[i]:
                    continue
                # calculate the score of the reassignment
                y_i_old = y[i]
                y[i] = j
                new_score = get_score(x, y)
                if new_score <= score:
                    score = new_score
                else:
                    y[i] = y_i_old
            # plot the reassignments if that's enabled
            if plot and i % 10 == 0:
                for m in range(0, n_clusters):
                    color[y==m] = colors[m]
                plot(x, color, title='Point: {}, Score: {}'.format(i, score), block=False)
        # plot right after the iteration
        for m in range(0, n_clusters):
            color[y==m] = colors[m]
        plot(x, color, title='Iteration: {}, Score: {}'.format(k, score)) 

# calculate
if calculate:
    improve_score(x, y)


# plot
if plot:
    plt.show()
