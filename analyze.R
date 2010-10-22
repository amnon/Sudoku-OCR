# some experiments with the data in R

# read data
data <- read.csv("vectors.txt")
labels <- data[,1]
samples <- data[,-1]

# show a vector as an image
show <- function(vec) {
    dim(vec) <- c(44,44)
    # reverse image (in OpenCV, row order is top to bottom)
    image(vec[,ncol(vec):1])
}

# counts of each label (histogram)
print(table(labels))

# split set into training and test.
# try to make training set contain a uniform amount of samples of each class
perm <- sample(length(labels))
samples <- samples[perm,]
labels <- labels[perm]

n <- 100
train <- samples[1:n,]
trainLabels <- labels[1:n]
test <- samples[-(1:n),]
testLabels <- labels[-(1:n)]

# try to predict using Nearest Neighbor
predict <- knn1(train, test, trainLabels)
print(predict == testLabels)
print(mean(predict == testLabels))

# PCA
# TODO: to center or not to center?
if (FALSE) {
    pcdat = prcomp(t(samples), center=FALSE)
    show(pcdat$x[,1])
}
