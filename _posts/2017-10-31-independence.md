---
layout: post
title: "Independence"
category: posts
---

## Independence

What's the difference between "linear correlation" or "correlation", and "statistical dependence"? Succintly, "linear correlation" is a much more specific relationship between two variables than "statistically dependent". In words, linear correlation means that as one variable goes up by some amount, the other one goes up by another amount, and the ratio between how much both variables change is constant across the data set. Having stated it this way, you can see how there could be other kinds of correlation--both variables could again change together, but the ratio between how much the two variables change could itself change over the data set. As long as there's some pattern to how this ratio behaves over the data set, correlation of some kind will be present.

Mathematically, the covariance between the two variables is a measure of their linear correlation (to get the actual correlation coefficient, which is bounded by $$[0,1]$$, you'd have to divide the covariance by the standard deviations of both variables). If our two variables are $$x$$ and $$y$$, and we've centered both of them so they have zero mean, the covariance is

\begin{equation}
\mathrm{Cov}(x,y) = \langle xy\rangle = \frac{1}{n} \sum_{i=1}^{n} x_i y_i.
\end{equation}

Notice the structure of this expression: inside the $$\langle  \rangle$$, it's _linear_ in both variables. You could have other expressions like $$ \langle x^2y \rangle$$ or $$ \langle x^2 y^3 \rangle$$ that could be non-zero even as the covariance $$ \langle xy \rangle$$ is 0. As an example, consider the data for two variables shown in the picture.

![quadratic_correlation](../../../../images/quad_corr.png)

The data points for $$x$$ form a line, and the data points for $$y$$ form a parabola. The two variables are not linearly correlated, because on the left, $$y$$ goes _down_ as $$x$$ goes up, but on the right, $$y$$ goes _up_ as $$x$$ goes up.If you calculate $$ \langle xy \rangle$$, the first half of the terms $$\sum_{i=1}^{n} x_i y_i$$ would cancel the second half of the terms. But the expression 

\begin{equation}
\langle x^2y \rangle = \frac{1}{n} \sum_{i=1}^n x_i^2 y_i
\end{equation}

would not be zero. An expression like this is called a _higher-order correlation_, since there are terms of higher order than linear inside the $$ \langle  \rangle$$. 

Getting back to the main point, you can see now why "linear correlation" is a very specific relationship: it specifies not just that two variables are related, but a certain way those variables are related. A quadratic correlation, like equation above, specifies a different way those variables are related. But in both cases, a relationship exists between the two variables; put another way, in both cases, by observing one variable we get _information_ about the other variable (although we'd have to know how they're related in order to use this information). This is finally where the term "statistical dependence" comes in. As long as we can get information about one variable by observing another variable, we say the two are statistically dependent. This doesn't necessarily imply that the variables are linearly correlated, or quadratically correlated, but it does mean that correlations of some kind are present (some expression that looks like $$ \langle x^ny^n \rangle$$ is non-zero (is this true, what about log correlations?)). 

Flipping our terms around, we can see that statistical independence is a much stronger requirement than linearly uncorrelated variables--it implies that correlations of all kinds are 0, whereas "linearly uncorrelated" just means that the expression $$ \langle xy \rangle$$ is 0, and says nothing about higher-order expressions. This is why we specify "statistical independence" in our criterion for ICA, instead of just "uncorrelated". (What happens if you merely stipulate that the transformed variables are linearly uncorrelated? You get PCA!)
