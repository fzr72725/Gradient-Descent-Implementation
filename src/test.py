import logistic_regression as f
from gradientdescent import GradientDescent
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_classes=2,
                            random_state=0)

gd = GradientDescent(f.cost_batch, f.gradient_batch, f.predict, fit_intercept = True)
gd.fit(X, y)
print "batch fit coeffs: ", gd.coeffs
predictions = gd.predict(X)
print "batch fit accuracy:", sum(predictions==y)*1./len(y)

'''
gd1 = GradientDescent(f.cost_SGD, f.gradient_SGD, f.predict, fit_intercept = True, alpha=1)
gd1.fit_SGD(X, y)
print "sgd fit coeffs: ", gd1.coeffs
predictions1 = gd1.predict(X)
print "sgd fit accuracy:", sum(predictions1==y)*1./len(y)
'''