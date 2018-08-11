---
layout: post
title: "Power Analysis"
category: posts
---

### Power Analysis

In statistics, power analysis refers to calculating how many runs of an
experiment you need to perform if you want to detect an effect, given a lower
bound for the strength of the effect. For example, if you were doing an
experiment to detect the efficacy of a new drug, and you suspect that the drug
will only work in 1 out of 10 people (but when it works, it completely
eliminates the disease), how many patients will you have to test to conclude
that the effect of the drug is statistically significant?

You can also run the analysis backwards: if you know you have a fixed number of
subjects for your experiment, you can establish a lower bound for the size of
any statistically significant effects you might measure. For example, if you're
only doing 50 trials, then you can use power analysis to figure out that you'll
only be able to detect a condition that affects 1 out of 20 people. If the
effect you're looking is real, but only affects 1 out of 30 people, then you
won't be able to find statistically significant evidence for it. (Numbers here
obviously made up but that's the kind of statements this analysis lets you
make.)

#### Inputs and outputs

Instead of medical experiments, let's confine ourselves to finding out whether a
coin is biased or not [1]. To fully specify the problem, let's state precisely
what we want to figure out. The question we're going to answer is,

>If a coin's bias is $$x$$, and we do an experiment recording $$N$$ coin flips,
what's the chance that we find statistically significant evidence that the coin
is biased?

If we answer this question, we'll be able to figure out how many trials of
flipping the coin we need--just keep increasing $$N$$ until we calculate that we
have a very high probability of detecting the bias.

We've also specified that we're looking for _statistically significant_ evidence
that the coin is biased. This means that we need to select a level of
significance. I'm going to use the typical value of 0.05--that is, if we observe
a $$p$$-value under 0.05 in our experiment, then we conclude that the coin is
biased. Technically, the significance level is another parameter you'd have to
supply when doing power analysis, but for the rest of the post I'm going to fix
it at 0.05.

So, an answer to the question would be a function or expression that you could
supply with the suspected bias of the coin, $$x$$, and the number of flips in your
experiment, $$N$$, and receive an output of a probability. Let's figure out that
function.

#### Math

We still need to state what we mean by "a coin's bias is $$x$$". There's more than
one way to define this, but for this post $$x$$ will be the probability that the
coin lands on heads when you flip it. So an unbiased coin has $$x=0.5$$, but any
other value of $$x$$ means that a coin is biased.

We know what the probability of observing heads is when we flip the coin once.
What's the probability of observing $$h$$ heads when we flip the coin $$N$$ times?
This is the binomial distribution

\begin{equation}
P(h) = x^h \, (1 - x)^{N - h} \frac{N!}{(N - h)!\,h!}
\end{equation}

Let's write a function which we can supply with $$h$$ (and $$x$$ and $$N$$, of course)
that'll give us this probability:


{% highlight python %}
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
%matplotlib inline
{% endhighlight %}


{% highlight python %}
def Pheads(h, N, x):
    heads = h
    tails = N - heads
    return (x**heads) * ((1 - x)**tails) * comb(N, heads)
{% endhighlight %}

A quick check that adding up the probabilities for all the outcomes we can
observe is 1:


{% highlight python %}
a = np.arange(101)
f = [Pheads(A, 100, 0.8) for A in a]
np.trapz(f, a)
{% endhighlight %}




    0.99999999989814803



We're good. Now we can calculate the outcome of doing an experiment.

The next question to answer is, given a certain outcome, what can we conclude
about the coin? Now we're going to start the statistics--we conclude that the
coin is biased if the $$p$$-value is below 0.05. So we just need a way to
calculate the $$p$$-value from the outcome of the experiment.

To do that, assume that the coin is actually fair (this corresponds to our null
hypothesis). We need to calculate the probability of the outcome we observe _or
one more extreme_ occuring. In our context, say we observe 3 heads after 10 coin
flips. We need to find the probability of getting 3 heads if the coin was fair.
We also need to get the probabilities of observing all events more extreme than
3 heads, which means getting 2, 1, or 0 heads. The sum of these probabilities is
the $$p$$-value.

On the other hand, another way to phrase this outcome is: the difference between
the expected number of heads (5) and the observed number was 2. Put this way,
the more extreme outcomes correspond to an observed difference of 3, 4, and 5,
which means we'd have to also add up the probabilities for observing 8 heads, 9
heads, and 10 heads. Which way is right?

They both are. The first way of adding up probabilities corresponds to a one-
tailed $$p$$-value, and the second corresponds to a two-tailed $$p$$-value (since
we're adding up probabilities on both sides/tails of the distribution). If we
knew that the coin was biased to land on tails more often, then we'd use first
calculation, where we add up the probabilities for landing on fewer and fewer
heads. But we've stated only that we suspect the coin to be biased; we don't
know if it's biased towards landing on heads more, or on tails more (we're not
testing if $$x > 0.5$$ or $$x < 0.5$$; we're just making sure that $$x\neq 0.5$$. This
means that we add up the probabilities all outcomes where the difference from
$$x$$ and 0.5 is greater than what we observe, or the two-tailed $$p$$-value.


{% highlight python %}
def pvalue(h, N):
    h = min(h, N - h)
    left_tail = sum(Pheads(i, N, 0.5) for i in xrange(h + 1))
    right_tail = sum(Pheads(i, N, 0.5) for i in xrange(N - h, N + 1))
    if (2 * h == N):
        right_tail -= Pheads(h, N, 0.5)
    return left_tail + right_tail
{% endhighlight %}

Let's test this out: if we flip the coin 10 times and observe 4 heads, we
shouldn't have much reason to suspect that the coin is biased. What's the
$$p$$-value that corresponds to this outcome?


{% highlight python %}
pvalue(4, 10)
{% endhighlight %}




    0.75390625



It's a very large number, and certainly not less than 0.05, which matches our
intuition that we have very little reason to suspect a bias. On the other hand,
what's the $$p$$-value for observing 1 head when flipping the coin 10 times?


{% highlight python %}
pvalue(1, 10)
{% endhighlight %}




    0.021484375



This $$p$$-value is 0.02, which is less than 0.05. That means that at this
significance level, we'd conclude that the coin is biased. This also matches our
intuition where we'd be suspicious if we only saw one head after 10 flips.

Now that we can calculate the $$p$$-value, and therefore know how to draw a
conclusion from every experiment, we can answer our original question: What is
the probability that we conclude the coin is biased? Let's write an expression
for that, including all our variables:

\begin{equation}
P(\mathrm{biased}, h, N, x)
\end{equation}

We're going to externally fix $$N$$ and $$x$$--in other words, they'll be parameters
we supply to the function that calculates $$P$$. Let's leave them out of the
expression. That means there's only one variable that we need to expand this in,
$$h$$:

\begin{equation}
P(\mathrm{biased}) = \sum_{H=1}^N P(\mathrm{biased} \mid h \;\mathrm{is}\; H) \;
P(h \;\mathrm{is}\; H)
\end{equation}

The binomial formula, encoded in the function Pheads we wrote above, gives us
$$P(h \;\mathrm{is}\; H)$$. To figure out the other factor, we know that we only
ever conclude the coin is biased if the $$p$$-value is less than 0.05. This means
that whenever $$H$$ is a value close enough to 0.5 such that the $$p$$-value is
above 0.05, there's 0 probability we conclude the coin is biased, and so
$$P(\mathrm{biased} \mid h \;\mathrm{is}\; H) = 0$$. Whenever $$H$$ is sufficiently
close to 0 or $$N$$ such that the $$p$$-value is under 0.05, the probability of
concluding the coin is biased is 1. Therefore the factor $$P(\mathrm{biased} \mid
h \;\mathrm{is}\; H)$$ has the effect of filtering out all terms in the sum for
which the $$p$$-value is not below 0.05.

Based on this intuition, the Python function that calculates
$$P(\mathrm{biased})$$, which we'll call the _power_, is:


{% highlight python %}
def power(x, N):
    return sum(Pheads(h, N, x) for h in xrange(N + 1) if pvalue(h, N) < 0.05)
{% endhighlight %}

#### Giving it a spin

If the coin isn't actually biased, and we observe 50 coin flips, we should
almost never see evidence that would lead us to conclude the coin is rigged. The
probabilty of making such a conclusion, the power, should be low:


{% highlight python %}
x = 0.5
N = 50
prob = power(x, N)
print('probability of concluding rigged: {0}'.format(prob))
{% endhighlight %}

    probability of concluding rigged: 0.0328391375643


There's a 3.2% chance we'd conclude that the coin is rigged. This is a small
number, like we suspected, so the code seems like it's working. I'm actually
surprised that the probability is this high, though--50 coin flips is a lot. But
remember that we're basing this calculation on a significance level of 0.05, and
that we we made this test more stringent, by lowering 0.05 to say 0.01, we'd
calculate smaller powers.

For the opposite scenario, where the coin strongly biased and we do 50 flips,
the power is


{% highlight python %}
x = 0.75
N = 50
prob = power(x, N)
print('probability of concluding rigged: {0}'.format(prob))
{% endhighlight %}

    probability of concluding rigged: 0.944876640866


So we'd almost certainly be able to see evidence for a bias by doing 50 flips,
if the bias is as strong as a $$3/4$$ chance to land on one side of the coin.

#### Plots with fixed N

Since we're now reasonably sure our code is correct, let's get a better sense of
the behavior of this function by making plots. We'll first look at the power
across all possible values of the bias, for different $$N$$.


{% highlight python %}
from collections import OrderedDict

X = np.linspace(0, 1, 100)
N = [10, 40, 100]
power_curves = []
for n in N:
    power_curves.append([power(x, n) for x in X])
curve_dict = OrderedDict(zip(N, power_curves))
{% endhighlight %}


{% highlight python %}
for k in curve_dict:
    plt.plot(X, curve_dict[k], label=r'$$N = {0}$$'.format(k))
plt.legend(loc='lower left')
plt.xlabel(r'$$x$$', size=20, usetex=True)
plt.ylabel('power', size=16)
plt.axis([0, 1, 0, 1])
{% endhighlight %}




    [0, 1, 0, 1]




![png](data:image/png;base64,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
){:width="100%"}


The first thing we notice is that these curves are symmetric across $$x=0.5$$.
This makes sense since we're not testing for whether our coin is biased
specifically towards heads or tails, but just whether it's biased in some way.
So a bias of landing on heads 90% of the time and a bias of landing on tails 90%
of the time are treated equally.

The next interesting detail is that the curves start out very fat, for $$N = 10$$,
and get sharper and sharper as $$N$$ increases. That just means that as we do more
coin flips in our experiment, we're able to detect more subtle biases, i.e. when
$$x$$ is closer to being 0.5. Let's make this same plot again for larger values of
$$N$$.


{% highlight python %}
X = np.linspace(0, 1, 100)
N = [100, 300, 1000]
power_curves = []
for n in N:
    power_curves.append([power(x, n) for x in X])
curve_dict = OrderedDict(zip(N, power_curves))
{% endhighlight %}


{% highlight python %}
for k in curve_dict:
    plt.plot(X, curve_dict[k], label=r'$$N = {0}$$'.format(k))
plt.legend(loc='lower left')
plt.xlabel(r'$$x$$', size=20, usetex=True)
plt.ylabel('power', size=16)
plt.axis([0, 1, 0, 1])
{% endhighlight %}




    [0, 1, 0, 1]




![png](data:image/png;base64,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
bwAAAABJRU5ErkJggg==
){:width="100%"}


The curves continue to get narrower as we increase $$N$$. Although I'm still a
little surprised that the curve for $$N=1000$$ doesn't look like an upside-down
delta function; this plot implies that if you suspect a bias where $$x=0.55$$, you
still can't be absolutely sure you've done enough testing to catch it _after
1000 coin flips_!

There's something else that's surprising in the first plot above. Look at the
lowest points on each of the curves, right in the middle where $$x=0.5$$. This
point doesn't always get higher as you increase $$N$$! When I first took a glance
at these plots, I came away with the pattern that "as you increase $$N$$, the
curves get skinnier and the nadir gets higher." But in the first plot, the nadir
is highest for the curve where $$N=40$$, which is not the skinniest or the fattest
curve! That means the relationship between the height of the nadir and the width
of the power curve, or $$N$$, may not be as simple as we might think.

#### Plots with fixed x

Let's plot the value of the power when $$x=0.5$$ for different values of $$N$$. You
can interpret this value as the probability that you'd conclude that you had a
rigged coin, when in fact the coin wasn't biased at all.


{% highlight python %}
N = [10, 30, 60, 90, 150, 250, 400, 700, 1000]
z = [power(0.5, n) for n in N]
{% endhighlight %}


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$$x = 0.5$$')
plt.xlabel(r'$$N$$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
{% endhighlight %}




    <matplotlib.legend.Legend at 0x106ae3450>




![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZcAAAETCAYAAAD6R0vDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xl8lOW58PHflT0kgUASUBJWQQREFiNqFawbYLWC1LX2
1PPWitZaa231wKm1attTfW3dqsejr/a8Pb7n1AUQ0SJUBevSikLCjkAAlQSErAiZSSaZXO8f80yY
hMkyk8lMMrm+n08+zNzPM8/cz4zmyr1dt6gqxhhjTCQlxLoCxhhj4o8FF2OMMRFnwcUYY0zEWXAx
xhgTcRZcjDHGRJwFF2OMMRFnwcUYY0zEWXAxxhgTcRZcjDHGRFxSrCsQK7m5uTpy5MhYV8MYY3qV
9evXV6hqXkfn9dngMnLkSNatWxfrahhjTK8iIp935jzrFjPGGBNxFlyMMcZEnAUXY4wxERf14CIi
c0Rkh4iUiMjCIMdTReQl5/haERnplI8UEbeIbHB+/iPgNaeLyGbnNU+IiETvjowxxrQW1eAiIonA
U8AlwATgOhGZ0Oq0G4FqVR0DPAo8FHBst6pOcX5uCSh/GrgJGOv8zOmuezDGGNOxaLdcpgMlqrpH
VT3Ai8DcVufMBf7kPF4MXNheS0RETgT6q+pH6tv57L+AeZGvuumsZcVlnPPgakYt/AvnPLiaZcVl
sa6SMSbKoh1c8oF9Ac9LnbKg56hqI3AYyHGOjRKRYhH5m4jMCDi/tINrAiAiC0RknYisKy8v79qd
mKCWFZexaOlmymrcKFBW42bR0s0WYIzpY3rTgP4BYLiqTgXuBP5HRPqHcgFVfVZVC1W1MC+vwzVA
JgwPr9qBu8Hboszd4OXhVTtiVCNjTCxEO7iUAcMCnhc4ZUHPEZEkYABQqar1qloJoKrrgd3Ayc75
BR1c00TJ/hp3SOXGmPgU7eDyCTBWREaJSApwLbC81TnLgRucx1cCq1VVRSTPmRCAiIzGN3C/R1UP
AF+JyFnO2Mx3gdeicTPmeEOz09ooT49yTYwxsRTV4OKModwGrAK2Ay+r6lYReUBELndOex7IEZES
fN1f/unKM4FNIrIB30D/Lapa5Ry7FXgOKMHXonkzKjdkjjN3yvHDXalJCdw1e1wMamOMiZWo5xZT
1RXAilZl9wY8rgOuCvK6JcCSNq65Djg1sjU14Vj3eTX90xLJTE3mwOE6RKB/ehKzJ54Q66oZY6Ko
Nw3omx7uH7sr+XhvFT+5eBx/X3Qhex+8lBduPJPyIx4efHN7rKtnjIkiCy4mYh57eyd5WalcN314
c9k5Y3L53jmj+NM/Pue9nTb925i+woKLiYiP9lSydm8Vt5x3EmnJiS2O3T1nHGMHZ3LX4o3UuDwx
qqExJposuJiIePztXeRlpXL9mcOPO5aWnMij10yh8qiHny/bgi+RgjEmnllwMV328d4q/rGnkptn
jj6u1eJ3av4AfnLxyfxl0wGWb9wf5RoaY6LNgovpssff2UluZirXnzmi3fNunjma00cM5J5lW2xR
pTFxzoKL6ZJ1n1XxYYmv1ZKeErzV4peUmMAjV0/G26T87JWNNDVZ95gx8cqCi+mSx9/ZRU5GCtef
dfxYSzAjcjK497IJ/H13Jf/598+6t3LGmJix4GLCtv7zKt7fVcGCmaPpl9L59bjXnDGMi8YP5qGV
n7Lz4JFurKExJlYsuJiwPfb2LgZlpPBPZ7c/1tKaiPDb+aeRlZrEHS9uwNPY1E01NMbEigUXE5ai
L6rDarX45WWl8tv5k9h24Csef2dnN9TQGBNLFlxMWB73t1rOCq3VEmjWxBO4urCAp9/dzbrPqjp+
gTGm17DgYkK2YV8Nf9tZzvdnjCIjtWu5T+/95kTyB6Zz58sbOVrfGKEaGmNizYKLCdnjb+8ku18y
3z17ZJevlZmaxCNXT2FftYtfv7Gt65UzxvQIFlxMSDbuq2HNjnJumjGazC62WvzOGDmIW847iRc/
2cdb2w5G5JrGmNiy4GJC8vg7uxiQnsx3Q5wh1pGfXHQy40/sz8Ilm6g4Wh/Raxtjos+Ci+m0TaU1
rP70EN8/dxRZackRvXZKUgKPXTOFI/WNLFyy2ZJbGtPLWXAxnfaE02q54ZyR3XL9cSdkcffscby9
/SAvr9vXLe9hTF+1rLiMcx5czaiFf+GcB1ezrLisW9/PgovplC1lh3l7+yFuPHcU/SPcagn0vXNG
cfboHB54fRtfVLq67X2M6UuWFZexaOlmymrcKFBW42bR0s3dGmAsuJhOefydXfRPS+Kfu6nV4peQ
IPzu6skkiHDnyxvwWnJLY0Kmqhypa2BP+VE+3lvF/a9vxd3gbXGOu8HLw6t2dFsdIjPdx8S1rfsP
89a2g9xx0dhubbX45Wen88C8ifzkpY08895ubv36mG5/T2N6A5enkYojHsqP1lF+pJ7yox7Kj9RT
cbS+xb/lR+qp70Rape7c+sKCi+nQE+/sIistif91zqiovee8Kfm8ve0Qj761k5lj8zg1f0DU3tuY
aKpr8FJxtJ4KJ1AEDRZH66k4Uk+tx3vc60UgJyOF3MxU8rJSGZmTQV5WKnmZqeRmpZCXmcadL2/g
0JHjZ2EOzU7vtvuy4GLatW3/V6zaepDbLxzLgPTub7X4iQi/nncqn3xWxU9e2sDrPzq3zV0ujelp
GrxNVLZqVZS3ChT+50fqgmemyO6X7AsQmalMLshuDh55WankZqY0B5BBGSkkJbY/wvGv3xjPoqWb
W3SNpScnctfscRG970AWXEy7nnhnF1mpSdwYxVaL38CMFB6+ajI3/PFjHl61g19cNiHqdTDGz9uk
VNbWO91SLQNE65ZGtash6DWyUpN8wSErlfEn9Gfm2IBAkZXaHEByMlJJSYrckPi8qfkAPLxqB/tr
3AzNTueu2eOay7uDBRfTpu0HvmLl1i+5/YIxDOgXvVZLoPNOzuO7Z4/g+Q/2csEpgzlnTG5M6mHi
U1OTUuNuaHPc4ljw8FBVW0+w+SXpyYnNwWF0bibTRw0iLzPN6ZJqGTRi2fqeNzW/W4NJaxZcTJv+
sHoXmalJfO/c6LdaAi26ZDwf7KrgZ69sZOUdM6PaPWd6H1XlK3fjca2KwNaGv6zyqIfGIBEjJSnB
GbNIpWBgP6YOH0ie08Jo2T2V2uXkrfEq6p+KiMwBHgcSgedU9cFWx1OB/wJOByqBa1T1s4Djw4Ft
wH2q+jun7DPgCOAFGlW1sPvvJL7t+PIIKzZ/yW3njyG7X0pM65Keksij10xh/tN/577lW3n0mikx
rY+JjGXFZZ3uplFVaj3eNlsYFQEtjPIj9Xi8x8+USkqQY4EhM5UJJ/ZvESwCg0ZWahIi0t0fQVyL
anARkUTgKeBioBT4RESWq2pgOtwbgWpVHSMi1wIPAdcEHH8EeDPI5c9X1Ypuqnqf88TqXWSkJHJj
jFstfpOHZfOjC8bw2Nu7uHD8YC47bWisq2S6wL+ozz/AXFbj5u7FG/mgpIKCgektg8ZR3zhH63Ua
AAkCOZnHAsNJgzObg8exGVO+fwekJ5OQYAEjWqLdcpkOlKjqHgAReRGYi68l4jcXuM95vBh4UkRE
VVVE5gF7gdroVbnv2XXwCCs2H+AH553EwIzYtloC/fD8MazZUc7PX91C4YhBnDAgLdZVMmFQVX6z
YvtxwcLjVRavLwVgUEZK81Ta04cPPK4ryv/voIwUEi1g9EjRDi75QGDSqFLgzLbOUdVGETkM5IhI
HfAv+Fo9P2v1GgX+KiIKPKOqzwZ7cxFZACwAGD58eBdvpXuE0lXQXZ5YXUJ6ciLfnzE6qu/bkeTE
BB69ejLfeOJ97lq8kf/63nTruuhFDn5Vx7LiMpYWlVEeZM0FgAA7f3MJyR1MrTU9X28aiboPeFRV
jwb5hXKuqpaJyGDgLRH5VFXfa32SE3SeBSgsLOxxeUWCdRUsWroZIGoBZtfBI7yxaT+3nHcSg3pQ
q8VvdF4mP790Ar9YtoUXPvo8IhuWme7j8jTy160HWVJUyoclFTQpTBuezYD0ZA67j5+uOzQ73QJL
nIh2cCkDhgU8L3DKgp1TKiJJwAB8A/tnAleKyP8GsoEmEalT1SdVtQxAVQ+JyKv4ut+OCy493cOr
drSZ/ydaweUPTqvlph7Wagn0nTOH8/a2g/zbiu187aRcxgzOjHWVTICmJmXt3iqWFpWyYvMBaj1e
8rPTue38MVwxrYBRuRnH/SEF3b+oz0RXtIPLJ8BYERmFL4hcC3y71TnLgRuAfwBXAqvVt7nHDP8J
InIfcFRVnxSRDCBBVY84j2cBD3T7nXSDtvL8dGf+n0Alh47y+qb9LJg5uke2WvxEhIevPI1Zj73H
nS9vYMkPvmZ/7fYAe8qPsrSojFeLyyircZOZmsSlp53I/GkFTB85qMVgeiwW9ZnoimpwccZQbgNW
4ZuK/EdV3SoiDwDrVHU58DzwgoiUAFX4AlB7hgCvOl1lScD/qOrKbruJbjQ0O52yIIGkO/P/BHpy
9S7SkhJZ0INbLX6D+6fx2ysm8YP/LuIPq0u48+KTY12lPqnG5eH1TQdYWlRK8Rc1JAjMGJvH3XPG
MWvCCaSntL1oMNqL+kx0RX3MRVVXACtald0b8LgOuKqDa9wX8HgPMDmytYyNu2aP467FG2nwHhsO
Sk1KiEpXwZ7yoyzfuJ/vzxhNTmZqt79fJFwy6UTmT83nqTUlnD8uj6nDB8a6Sn2Cp7GJv+0sZ2lR
Ke9sP4TH28S4IVn86zdOYe6UfIb0t1l8pncN6Me9eVPzeXPzflZtO4Tgy3baPz2J2RNP6Pb3fnJ1
CSlJCT16rCWY++ZOZO3eKu58eSN/uf1c+qXYf9LdQVXZUvYVS4pKWb5xP1W1HnIzU/jOWSOYPy2f
iUP728w904L9n9jDDO6fzsB+yRTfO4sPSyq4/rm1/GbFNn49b1K3vefeilqWbSjje+eMIi+rd7Ra
/PqnJfO7qybz7ec+4t9WbO/Wz6kvOnDYzbLi/SwtKmXXoaOkJCVw8YQhfGtaPjPG5tlYl2mTBZce
psrlYaCTbuWcMbncNGMU/+f9vZw/bjAXjh/SLe/5h9W7SElKYMF5vavV4nf2STl8/1zf53Th+CGc
P25wrKvUq7k8jaza+iVL1pfx4e4KVKFwxED+7YpJXDrpxJglMTW9iwWXHqbG5WmxKv5ns8fxQUkl
dy/exMo7Zka8ZfFZRS2vbdjPP39tJIOzem9f+U9njeO9nRXcvXgTq+6Y2aNnu/VETU3KR3sqWVJU
xptbDuDyeCkYmM6PLhjL/Kn5jMzNiHUVTS9jwaWHqaptID/72C/51KREHr92Ct/8wwfcvXgjf/zn
MyLat/3kmhKSEoSbe2mrxS8t2Zfccu5TH/CvSzfz9Hem2RhAJ5QcOsqrxaW8WlTG/sN1ZKUmcfnk
ocyfVkDhiIGWi8uEzYJLD1Pj8nDq0P4tyk4eksWiS07hvte38f8++px/itCq9M8ra3m1uIzvnj2i
V7da/CYM7c9PZ43jwTc/ZWlRGd86vSDWVeqRqms9vL5pP0uKyti4zzd9eObJeSz8xnhmTRhiO36a
iLDg0sNU1XqCdunc8LWRrNlRzq//sp2zT8phzOCsLr/XU2tKSEwQbjnvpC5fq6e4acZoVm8/xC+X
b+XM0YMoGNgv1lXqETyNTazZcYilRaWs/vQQDV7llBOyuOfS8Vw+eSiDbfqwiTALLhEQqWSTbo+X
+samoPun+Felz3n8fX784gZevfWcLm2Duq/KxdKiMr5z1oi4WpeQmCD8/urJzHnsPX768kb+fNNZ
fbZrR1XZVHqYJUWlvL5xP9WuBnIzU7nh7JHMn1bAhFYtZGMiyYJLF0Uy2WSVywPAoIzgs3EG90/j
wfmTWPDCeh55aycLLzkl7Ho/taaEBImvVovfsEH9+OXlE7l78Sae/2AvN83s3eNJodpf4+bV4jKW
FpWyu7yWlKQEZk0YwremFTBjbC5JNn3YRIEFly6KZLLJ6lpfcGlv58dZE0/guunDeOa93Zx3ch5n
n5QTcp33VblYvL6U688cHrd7olx1egFvbzvIw6t2MOPkXE45Ib7/Sq+tb2Tlli9ZUlTKP/ZUogpn
jBzITTNGc8mkE21raBN1Fly6KJLJJqubWy7tT6P9xWUT+GhPFT99eQNv/nhmyOsO/v1dp9Xy9fhr
tfiJCL+dP4nZj73HHS9u4LXbziE1Kb4Gqr1Nyj92V7K0qJQ3t3yJu8HL8EH9+PGFY7liaj4jcmz6
sIkdCy5dFMlkk9Uu3/4WAzsIFv1SknjM2VP+nte28MS1Uzo97ba02sUr60q5bvpwThwQnYSYsZKT
mcpD3zqNG/+0jkfe2smiS8bHukoRUXLoCEuKylhWXMaBw3VkpSUxb+pQvjWtgNNHDLQp2KZHsODS
RXfNHhexfSn83WID2+kW85s8LJs7LhzL79/ayQWn5HHF1M5Nu/33d3cjAj+I41ZLoAvHD+G66cN5
9r09XDBuMGeODr0bsSeoqvXw+sb9LCkqZVPpYRIThPNOzuPnl47novE2fdj0PBZcusg/rvKzVzbS
2KQM6Z/KokvGhzVbrMoJLp3tH7/1/DH8bWc59y7bSuGIQQwb1P6027IaN6+s28fVhcOilsa/J7jn
0vH8fXcFd768kZV3zCArrXeMP9Q3elnz6SGWFJWx5tNDNDYpE07szz2XjmfulPxelwfO9C02bSQC
5k3NZ7DzP/rT3zk97D0qalweBqQnd3o2T2KC8Og1U1Dgpy9vxNvU/s7NT79bAviCUl+SkZrEI1dP
4cBhN/e/vi3W1WmXqlL8RTW/WLaFM//tHW75f0Vs2FfD984dxZs/nsGKH8/g+zNGW2AxPZ61XCLE
3y1WddQT9jWqXA0djre0NmxQPx6YO5E7X97If/xtNz9sI3AcOOzm5U9KufL0YeT3oVaL3+kjBnLr
18fw5JoSLho/hDmndv82BqEorXaxrLiMpUVl7KmoJTUpgdkTT2D+tHzOHWPTh03vY8ElQlweX3Cp
rK0P+xqtk1Z21hVT81n96SEefWsnM8bmclpB9nHnPP3ubppU+eH5fWOsJZjbLxzLuzsP8a+vbmba
iOyYp7w5Wt/Im5sPsLSojH/sqQRg+qhB3Hyeb/pw/17SfWdMMBZcIqCpSalvbAKgoistl1pPWKvl
RYTfzJvE+s+ruePFDbzRatOsA4fdvPjxPq4qLOjT6VBSkhJ49OopXPaHD1i4ZDPP31AY9ZlV3ibl
77srWLK+lJVbv6SuoYmROf248+KTuWJqfofjZsb0FhZcIiBwplhlF4JLjash7MV+A/ol8/urJ3P9
c2v51Rvb+e38Y5tm/YfTarn1631rrCWYsUOyWHjJKdz/+jb+/PE+vn3m8Ki8786DR1hSVMqy4jIO
flVP/7Qk5k8r4FvT8pk23KYPm/hjwSUCWgSXLnSLVdV6Qh5zCfS1k3JZMGM0z7y3hwtOGczFE4Zw
8Ks6/vzJPr41rcD+KnbccPZI3tl+iF+9sY2vnZTTbXuVVB6tZ/nG/SwtKmNzmW/68Pnj8vjlNwu4
4JTBNn3YxDULLhHg9nS95VLX4MXd4A1rzCXQnbNO5v1dFdzxYjFZacl8+VUdACcPyezSdeNJQoLw
8FWnMfvR9/jJyxt45eazIzZgXt/o5Z3tvuzD7+4op7FJOTW/P/deNoHLpwwlN9NmeZm+wYJLBLgC
gkvF0fBaLv7UL51ZQNme1KRELp98Ig+u3EFtQL1+99ed5GSmhj1NOt6cOCCdX18xidv/XMzT7+7m
RxeODftaqkrRFzUsLSrljU0HOOxuYHBWKjeeO4r50woYd0LXt0cwprex4BIB/m6xwVmpzQshQ1Vd
60v90lZG5FC88NEXx5WFm0wznl0+eShvbzvI4+/s4uvjBjOpYEBIr99X5UwfLi5jb0UtackJzJl4
AvOnFXDOmFwS+2iqf2PAgktEuDyNABQMTGdT6WGamjTkPUQi1XKByCbTjHe/mnsqH++t4o6XivnL
7TM6HAc5UtfAm5t92YfX7q0C4KzRg/jB10/iklNP6DWr/43pbhZcIqDOabkMG9SPoi9q+Kquod20
+cE0B5cujrlAZJNpxrsB/ZJ5+KrT+KfnP+bBNz/lvssnHneOt0n5oKSCpUWlrHKmD4/KzeBns05m
3tT8Pj2925i2WHCJAP+YyzDnl0zFUU/owSWEpJUdiWQyzb5gxtg8/vlrI/m/f/+M1zfup6rWw9Ds
dL5z1nCqXQ0sKy7j0JF6BqQnc+XpBcyfVsDUYdk2fdiYdkQ9uIjIHOBxIBF4TlUfbHU8Ffgv4HSg
ErhGVT8LOD4c2Abcp6q/68w1u5t/tljBQF/LoPJoPWMGhzY7y59uP7sLU5H9/OMqkdh6ua+YOLQ/
AlQ6Qb6sxs1DK3cg+DIrf2taPheMHxx3e8IY012iGlxEJBF4CrgYKAU+EZHlqhqYTfBGoFpVx4jI
tcBDwDUBxx8B3gzxmt3KHdAtBsd+QYWiqtZDVloSyRGaEjtvar4FkxA89vYugqX9HDIgjeduKIx6
fYzp7aKdDW86UKKqe1TVA7wIzG11zlzgT87jxcCF4vQ/iMg8YC+wNcRrdqtgLZdQ1bg8EekSM+Fp
a7LDwcN1Ua6JMfEh2sElH9gX8LzUKQt6jqo2AoeBHBHJBP4FuD+MawIgIgtEZJ2IrCsvLw/7Jlrz
j7n4d3YMq+XiaojIYL4JT1uTHWwShDHh6U15vO8DHlXVo+FeQFWfVdVCVS3My8uLWMXcDV5SkxJI
SUpgYL/ksFbpV3cx9YvpmrtmjyO91TRkmwRhTPiiPaBfBgwLeF7glAU7p1REkoAB+Ab2zwSuFJH/
DWQDTSJSB6zvxDW7ldvjpV+K7xdTTmZqWPnFql0exoY4CcBEjk2CMCayoh1cPgHGisgofAHgWuDb
rc5ZDtwA/AO4ElitqgrM8J8gIvcBR1X1SScAdXTNbuXyeJv/6s3JSAkr7X51bejTl01k2SQIYyIn
qsFFVRtF5DZgFb5pw39U1a0i8gCwTlWXA88DL4hICVCFL1iEfM1uvZFW6hq8pDstl9zMVD798quQ
Xl/f6KXW441I6hdjjOkJor7ORVVXACtald0b8LgOuKqDa9zX0TWjyeVpbN6ca1BGSsgD+jXNa1ys
5WKMiQ+9aUC/x3I3BHSLZaZQ42qg0dvU6df7U78Mstlixpg4YcElAtyeY91iOc5+HVWuzrde/JmU
I7E63xhjegILLhEQOKCf67Q+QpmO7O8Ws5aLMSZedDq4iEiKiLwqIjO7s0K9kbuh5VRkCC24VEUw
aaUxxvQEnQ4uTmqVi0J5TV/h9nhJSzk25gKEtNalxmXdYsaY+BJqoPgQOKs7KtKbuRu89AtY5wKE
tNalqraBzNQky7hrjIkboU5F/imwTESOAsuAA9Aymayqdn6aVBxQVd9sMafl0j8tmaQECSl5ZY3L
Y60WY0xcCbXlshk4Cd/eKZ8DHqAh4Ce8DeR7sfrGJlRpDi4JCcKgjJTmcZTOqHJ5bDDfGBNXQm25
PABBt73os/wZkfsFJD3MyUwNqVus2hX6tsjGGNOThRRcWq+MN8c2CvO3XAByM1NCGtCvrvUwKsf2
YTfGxI+wZ36JSKaIjBCRPj1Y4PY0ApCecixO52SkhDQV2ZJWGmPiTcjBRUQuE5EifJt47QEmOeXP
iUhUsxH3BP5usfRW3WKdHdBv8DZxpL7RxlyMMXElpODibDP8GlCBb1dICTi8F1+q/D7Fv8Vxv4Bu
sUEZKdR6vM3H2uPPK2YbhRlj4kmoLZdfAv+pqrOAx1od2wKcGpFa9SIuZ8wlLbnlmAt0biGlP/WL
bXFsjIknoQaX8cBLzuPWs8aqgZwu16iXqQvScsnJcJJXdmI6sqV+McbEo1CDy1dAbhvHRgLlXapN
LxR8zKXzyStrXBZcjDHxJ9Tg8hawSESyA8pURFKB24A3I1azXsI/Fblfi6nIvpZLRScG9atq/d1i
NuZijIkfoS6i/DnwMbAD386PCiwETgMGAPMiWrtewD9oH7jO5Vjyyo5bLtXWcjHGxKGQWi6q+hkw
DXgDuBjwAjOBj4AzVXV/pCvY0zUvogzoFuuXkkRackKnpiNX13pIT05sMSHAGGN6u1BbLqhqKXBj
N9SlV3J5vKQkJpCU2DJO52SkdmrMpdrVYGtcjDFxJ9R1LpeISEZ3VaY3cnsaSUs+/mPMzUyhopPd
YpYR2RgTb0JtufwFaBCR9cAaYDXwoarWRbxmvYRvF8rjP8aczFQOHen4Y6m2jMjGmDgU6myxk4Hb
8aXbvxHf7LFqEfmbiPyyL26B7PJ4Wwzm+3U2v1h1rccG840xcSfUAf0SVX1GVa9T1RPwrci/C2gE
7sXXkulT6hq8LQbz/Xz5xTyotr9DQbWrwVK/GGPiTsgD+gAi0g+YAZwPXAhMxbfA8m+Rq1rv0F7L
xeMkpeyfFjx4NHqbOOxusNQvxpi4E+qA/gMi8gG+VC+LgcnAy8CZQI6q9r11Lg3eFgso/TqzSv+w
21lAad1ixpg4E+qYyz3AFOAJYLSqXqKqD6vqelVt6swFRGSOiOwQkRIRWRjkeKqIvOQcXysiI53y
6SKywfnZKCJXBLzmMxHZ7BxbF+I9dYnb03a3GNDuWpfmBZTWcjHGxJlQg8uPgb8C3wMOiMh6EXnY
maKc2dGLRSQReAq4BJgAXCciE1qddiNQrapjgEeBh5zyLUChqk4B5gDPiEhgt975qjpFVQtDvKcu
cTe03S0G7a/Sb079YmMuxpg4E+qA/h9UdT6+5JXTgf/Glyn5z0CViHzYwSWmAyWqukdVPcCLwNxW
58wF/uQ8XgxcKCKiqi5VbXTK0zg+K3NMuDzBu8Vym1subQcXS/1ijIlXYW1zrL4pUFuAIqAY+BTf
5ICzOnhpPrAv4HmpUxb0HCeYHMZJ5S8iZ4rIVmAzcEtAsFHgr05LakFbby4iC0RknYisKy+PTAJn
t8cbNHWLf+1Ku91itdYtZoyJTyHNFhORrwEX4JsldjaQClQC7+JrbayJcP1aUNW1wEQRGQ/8SUTe
dBZwnqv8yr79AAAWJElEQVSqZSIyGHhLRD5V1feCvP5Z4FmAwsLCLrd8VLXNAf2UpAT6pyW12y1W
7WwUNshaLsaYOBPqVOQPgBrgPXzZkNeo6uYQXl8GDAt4XuCUBTun1BlTGYAvgDVT1e0ichTfOpt1
qlrmlB8SkVfxdb8dF1wizeNtwtukQQf0wTeo317a/WqXh9SkhKBjNsYY05uFGlwKgWLtaGVg2z4B
xorIKHxB5Frg263OWQ7cAPwDuBJYrarqvGafqjaKyAjgFOAzJ9dZgqoecR7PAh4Is34hqfP4Jsil
B0n/Ah2v0q+utdQvxpj4FFJwUdUi/2NndthAfDO7jnby9Y0ichuwCkgE/qiqW0XkAXwtkOXA88AL
IlICVOELQADnAgtFpAFoAm5V1QoRGQ28KiL++/kfVV0Zyn2Fy9XgG/Jpu+WSwt6K2jZf70taacHF
GBN/Ql6hLyKzgd/gW+8i+HaiLAJ+rqpvdfR6VV2Bb6OxwLJ7Ax7XAVcFed0LwAtByvfgW8wZdf6N
woKNuYCvW2z959Vtvt6Xbt+mIRtj4k+oK/Rn48uMnAn8CrgV+DWQBawQkYsjXsMezBVkF8pAuRkp
VNV68DYF70WsrrWWizEmPoXacrkP3yLKywJX5DvdWm8A9+PLlNwn1AXZhTJQTmYqTQo1Lk/ziv1A
1S6PzRQzxsSlUNe5TAaeap3qxXn+7/i6yvoMVwfdYoPaWaXvbVJq3JYR2RgTn0INLvVA/zaOZTnH
+wx/cAm2iBKOJa8MNh35K3cDqraA0hgTn0INLu8Cv3KmBTcTkeH4usy6dRFlT+PvFmur5dJeCpgq
J/WLTUU2xsSjUMdc/gX4ENghIh8BB4AT8KV9qXGO9xkdDejntJMCpsYJLjagb4yJR6EmrtwJnIYv
5X4qMA1fEsnHgSmquiviNezB3P6WS3LwGJ3dL4UEgaogYy7+jMg2oG+MiUchr3NR1QPO7LBT8SWZ
LAM2q+qRSFeup3N7nEWUbbRcEhOEQRkpVAQJLv6kldk2oG+MiUPhLKK8F/gpvrUufkdF5GFV/XXE
atYLuBu8JCYIyYnS5jmDMlKCdotV25iLMSaOhZoV+X7gF8Bz+PZiOQgMAa4D7heRJFW9L9KV7Klc
Hi/9khNxUs8ElZOR2uaAfkpiQpuTAYwxpjcLteVyE/B7Vb0roGwrsFpEDgML8M0a6xPqGrykdRAc
cjJT2Lr/q+PKa2obGJiR3G5gMsaY3irUqcgD8CWdDGalc7zPaGsXykC5baTdr3J5bAdKY0zcCjW4
rAXOaOPYGc7xPsPl8baZ+sUvJyOFI3WNeBpbJDWgxoKLMSaOhdotdju+9PaNwCscG3O5GvgeMFdE
mgNW6zQx8aauwdvhRl/+nGJVtR5OGJDWXF5V62HcCVndWj9jjImVUIPLJuffB52fQIJvb3s/DeP6
vUqnWi4BKWACg0uNq8FaLsaYuBXqL/8H8AUNg28/l44ST+YESV7Z1KRUW7eYMSaOhboT5X3dVI9e
yd3gbXOLY7+c5vxixwb1j9Q10mRJK40xcSzUAX0TwO3xkp7c/kfo7xYLXOviT1pp6faNMfHKgksX
uDyN9Oug5ZKVmkRKYgIVtcdaLv7V+dZyMcbEKwsuXVDX0NTmXi5+IkJOZgpVAS0Xf14xG3MxxsQr
Cy5havQ24fE2dSp9S05mSosB/WqXZUQ2xsQ3Cy5hcjnp9juaigwwKCO1xYB+c8slw8ZcjDHxyYJL
mOo62CgsUG5GChWB3WIuD0kJQmZqXC8DMsb0YRZcwtS8C2UnWi6+brF6VH1LhKpdHgZmpFjSSmNM
3LLgEqbmXSg7NeaSSl1DU3NAqqr12DRkY0xcs+ASJlcI3WLNq/SdrrFqS/1ijIlzUQ8uIjJHRHaI
SImILAxyPFVEXnKOrxWRkU75dBHZ4PxsFJErOnvN7lAXwoB+rn+VvrPWpbrWUr8YY+JbVIOLiCQC
TwGXABOA60RkQqvTbgSqVXUM8CjwkFO+BShU1SnAHOAZEUnq5DUjzt9y6WgRJRzbyrhFy8UWUBpj
4li0Wy7TgRJV3aOqHnxbJc9tdc5c4E/O48XAhSIiqupS1UanPI1jCTQ7c82I84+5pKd0/BE2p4Bx
BvV9e7nYmIsxJn5FO7jkA/sCnpc6ZUHPcYLJYSAHQETOFJGt+FL73+Ic78w1cV6/QETWici68vLy
Lt2I2+OLcx0lrgTIyfB1i1Uc9XCkvpHGJm1uzRhjTDzqVQP6qrpWVSfi2/VykYikdfSaVq9/VlUL
VbUwLy+vS3UJZSpyekoiGSmJVB71NC+gzLYxF2NMHIt2cCkDhgU8L3DKgp4jIknAAKAy8ARV3Q4c
BU7t5DUjLpSpyOCbjlxZW38s9YutzjfGxLFoB5dPgLEiMkpEUoBrgeWtzlkO3OA8vhJYrarqvCYJ
QERGAKcAn3XymhHn9ngRgdSkzn2EOZkp1nIxxvQZUc0/oqqNInIbsApIBP6oqltF5AFgnaouB54H
XhCREqAKX7AAOBdYKCINQBNwq6pWAAS7Znffi9vZ4rizq+xzMlIoq6lrTrdvSSuNMfEs6smtVHUF
sKJV2b0Bj+uAq4K87gXghc5es7u5Gryd7hID36D+ptLDVFm6fWNMH9CrBvR7kjqPt8O9XALlZKZQ
VeuhqtZDYoKQlWZJK40x8cuCS5hcnhBbLpmpNDYpn1e6yE5PJiHBklYaY+KXBZcwuRu8nZqG7Jfr
LKQsOXTUVucbY+KeBZcwuT3eTiWt9PMvpNxbUWuD+caYuGfBJUyuhsaQWi7+FDAebxPZlvrFGBPn
LLiEye3xdipppV9OQFeYpX4xxsQ7Cy5hcoc4WyxwnMUWUBpj4p0FlzC5Q1znkpyY0NwdZqlfjDHx
zoJLmEKdigzHusas5WKMiXcWXMLQ1KTUNzaF1C0GvrUuYKlfjDHxz4JLGELNiAywrLiMTaU1ACxc
uollxd2euNkYY2LGgksYju1C2bngsqy4jEVLN1PX0AT4Ng1btHSzBRhjTNyy4BIGdwgbhQE8vGpH
c0BqvkaDl4dX7Yh43Ywxpiew4BKG5l0oO9ly2V/jDqncGGN6OwsuYQh1zGVodnpI5cYY09tZcAmD
y9MI0OnZYnfNHndcF1p6ciJ3zR4X8boZY0xPYJuKhKGuueXSuY9v3tR8wDf2sr/GzdDsdO6aPa65
3Bhj4o0FlzD4x1xCmYo8b2q+BRNjTJ9h3WJhCHW2mDHG9DUWXMIQ6joXY4zpayy4hMFaLsYY0z4L
LmFwWXAxxph2WXAJg7vBS2pSAgkJEuuqGGNMj2TBJQzuMNLtG2NMX2LBJQwuj9e6xIwxph0WXMJQ
1+C1mWLGGNOOqAcXEZkjIjtEpEREFgY5nioiLznH14rISKf8YhFZLyKbnX8vCHjNu841Nzg/g7vz
Hlyexk6vzjfGmL4oqr8hRSQReAq4GCgFPhGR5aq6LeC0G4FqVR0jItcCDwHXABXAN1V1v4icCqwC
Ape8X6+q66JxH+4G6xYzxpj2RLvlMh0oUdU9quoBXgTmtjpnLvAn5/Fi4EIREVUtVtX9TvlWIF1E
UqNS61bcHusWM8aY9kQ7uOQD+wKel9Ky9dHiHFVtBA4DOa3O+RZQpKr1AWX/6XSJ/UJEgs4RFpEF
IrJORNaVl5eHfRPWcjHGmPb1ugF9EZmIr6vs5oDi61V1EjDD+fmnYK9V1WdVtVBVC/Py8sKug8um
IhtjTLuiHVzKgGEBzwucsqDniEgSMACodJ4XAK8C31XV3f4XqGqZ8+8R4H/wdb91G7fHS5oFF2OM
aVO0g8snwFgRGSUiKcC1wPJW5ywHbnAeXwmsVlUVkWzgL8BCVf3Qf7KIJIlIrvM4GbgM2NKdN+Fu
8NLPusWMMaZNUQ0uzhjKbfhmem0HXlbVrSLygIhc7pz2PJAjIiXAnYB/uvJtwBjg3lZTjlOBVSKy
CdiAr+Xzf7rxHnxjLtZyMcaYNkV9sYaqrgBWtCq7N+BxHXBVkNf9Gvh1G5c9PZJ1bE99YxOqlm7f
GGPa0+sG9GOteRdK6xYzxpg2WXAJkW0UZowxHbPgEiK3pxGAdEv/YowxbbLgEiK3pwmwjcKMMaY9
9ud3iFxOy8UWURrT+zU0NFBaWkpdXV2sq9LjpKWlUVBQQHJyclivt+ASIpcz5pJmLRdjer3S0lKy
srIYOXIkbWSN6pNUlcrKSkpLSxk1alRY17BusRDV+WeLWcvFmF6vrq6OnJwcCyytiAg5OTldatFZ
cAmRfyqyjbkYEx8ssATX1c/FgkuI/FORreVijDFts+ASIrfH1rkYY0xHLLiEqHkRpXWLGWN6mJUr
VzJu3DjGjBnDgw8+2OZ5I0eOZNKkSUyZMoXCwsJuqYvNFguRy+MlJTGBpESLy8b0NcuKy3h41Q72
17gZmp3OXbPHMW9q6/0OY8Pr9fLDH/6Qt956i4KCAs444wwuv/xyJkyYEPT8NWvWkJub2231sd+Q
Iapr8JKWbB+bMX3NsuIyFi3dTFmNGwXKatwsWrqZZcWtt6QK3fnnn89bb70FwD333MOPfvSjkK/x
8ccfM2bMGEaPHk1KSgrXXnstr732WpfrFi5ruYTI5Wmkn6V+MSbu3P/6Vrbt/6rN48Vf1ODxNrUo
czd4uXvxJv788RdBXzNhaH9++c2JHb/3/fdz7733cujQIYqLi1m+vOU2VzNmzODIkSPHve53v/sd
F110EQBlZWUMG3ZsL8aCggLWrl0b9P1EhFmzZiEi3HzzzSxYsKDDOobKfkuGyOWxvVyM6YtaB5aO
ykMxc+ZMVJVHHnmEd999l8TElr9j3n///S6/R6APPviA/Px8Dh06xMUXX8wpp5zCzJkzI/oeFlxC
VNfgtcF8Y+JQRy2Mcx5cTVmN+7jy/Ox0Xrr57C699+bNmzlw4AA5OTlkZWUdd7wzLZf8/Hz27dvX
fKy0tJT8/ODjQf7ywYMHc8UVV/Dxxx9HPLjY4EEIlhWX8f6uCrYd+IpzHlwdkb5WY0zvcNfsccf9
YZmenMhds8d16boHDhzg+uuv57XXXiMzM5OVK1ced87777/Phg0bjvvxBxaAM844g127drF37148
Hg8vvvgil19++XHXqq2tbQ5UtbW1/PWvf+XUU0/t0j0EY8Glk/yDefWNviZwJAfzjDE937yp+fx2
/iTys9MRfC2W386f1KXZYi6Xi/nz5/P73/+e8ePH84tf/IL7778/rGslJSXx5JNPMnv2bMaPH8/V
V1/NxIm+1tg3vvEN9u/fD8DBgwc599xzmTx5MtOnT+fSSy9lzpw5Yd9DW0RVI37R3qCwsFDXrVvX
6fPbaxJ/uPCCSFbNGBMl27dvZ/z48bGuRo8V7PMRkfWq2uHiGGu5dNL+IIGlvXJjjOnLLLh00tDs
9JDKjTGmL7Pg0kndNZhnjDHxyKYid5J/0K6npn4wxoRHVS3tfhBdHY+34BKCeVPzLZgYE0fS0tKo
rKy0DcNa8e9EmZaWFvY1LLgYY/qsgoICSktLKS8vj3VVepy0tDQKCgrCfr0FF2NMn5WcnBz2HvGm
fTagb4wxJuIsuBhjjIk4Cy7GGGMirs+mfxGRcuDzEF6SC1R0U3V6qr54z9A377sv3jP0zfvu6j2P
UNW8jk7qs8ElVCKyrjP5dOJJX7xn6Jv33RfvGfrmfUfrnq1bzBhjTMRZcDHGGBNxFlw679lYVyAG
+uI9Q9+87754z9A37zsq92xjLsYYYyLOWi7GGGMizoJLB0RkjojsEJESEVkY6/pEkogME5E1IrJN
RLaKyI+d8kEi8paI7HL+HeiUi4g84XwWm0RkWmzvIHwikigixSLyhvN8lIisde7tJRFJccpTnecl
zvGRsax3V4hItogsFpFPRWS7iJwd79+1iPzE+W97i4j8WUTS4vG7FpE/isghEdkSUBbydysiNzjn
7xKRG7pSJwsu7RCRROAp4BJgAnCdiEyIba0iqhH4qapOAM4Cfujc30LgHVUdC7zjPAff5zDW+VkA
PB39KkfMj4HtAc8fAh5V1TFANXCjU34jUO2UP+qc11s9DqxU1VOAyfjuP26/axHJB24HClX1VCAR
uJb4/K7/LzCnVVlI362IDAJ+CZwJTAd+6Q9IYVFV+2njBzgbWBXwfBGwKNb16sb7fQ24GNgBnOiU
nQjscB4/A1wXcH7zeb3pByhw/me7AHgDEHyLypJaf+/AKuBs53GSc57E+h7CuOcBwN7WdY/n7xrI
B/YBg5zv7g1gdrx+18BIYEu43y1wHfBMQHmL80L9sZZL+/z/cfqVOmVxx+kCmAqsBYao6gHn0JfA
EOdxvHwejwF3A03O8xygRlUbneeB99V8z87xw875vc0ooBz4T6c78DkRySCOv2tVLQN+B3wBHMD3
3a0n/r9rv1C/24h+5xZcDCKSCSwB7lDVrwKPqe9PmLiZUigilwGHVHV9rOsSZUnANOBpVZ0K1HKs
mwSIy+96IDAXX2AdCmRwfNdRnxCL79aCS/vKgGEBzwucsrghIsn4Ast/q+pSp/igiJzoHD8ROOSU
x8PncQ5wuYh8BryIr2vscSBbRPz7GwXeV/M9O8cHAJXRrHCElAKlqrrWeb4YX7CJ5+/6ImCvqpar
agOwFN/3H+/ftV+o321Ev3MLLu37BBjrzC5JwTcYuDzGdYoY8e3r+jywXVUfCTi0HPDPFLkB31iM
v/y7zmyTs4DDAc3uXkFVF6lqgaqOxPd9rlbV64E1wJXOaa3v2f9ZXOmc3+v+ulfVL4F9IjLOKboQ
2EYcf9f4usPOEpF+zn/r/nuO6+86QKjf7SpglogMdFp9s5yy8MR6EKqn/wDfAHYCu4Gfx7o+Eb63
c/E1lTcBG5yfb+DrZ34H2AW8DQxyzhd8s+d2A5vxzcKJ+X104f6/DrzhPB4NfAyUAK8AqU55mvO8
xDk+Otb17sL9TgHWOd/3MmBgvH/XwP3Ap8AW4AUgNR6/a+DP+MaVGvC1Um8M57sFvufcfwnwv7pS
J1uhb4wxJuKsW8wYY0zEWXAxxhgTcRZcjDHGRJwFF2OMMRFnwcUYY0zEWXAxxhgTcRZcjDHGRJwF
F2OMMRFnwcWYGBORu53NnFREdgc5nh1wvFpEXolFPY0Jha3QN6YHcHYDXIQvp9XpqloU5JxXgJtU
tSba9TMmVNZyMaZnuAi4yXl8cxvnfGKBxfQWFlyM6SGcwPEsvq1nWxCRbMACi+k1LLgY07M8AyAi
rQPMRfgy2xrTK1hwMSbGnPGWIgBnrGUPx3eNjVbVPdGumzHhsuBiTOwV4ttnxe8ZYJqIjI5RfYzp
MgsuxsRedquB+medf/8FbLzF9E4WXIzpYZxAsxi42ikqxMZbTC9jwcWYGHK6voKNpTwDZIvIlcA0
G28xvY0FF2NiK+gsMFV9G19XWFtrXozp0Sy4GBNbrcdbAj2LL/gY0+tYcDEmRpyB+jPaOeUZ518b
bzG9jgUXY2LAyRO2F7hSRF5xAk0LzjjLs8HyjBnT01niSmOMMRFnLRdjjDERZ8HFGGNMxFlwMcYY
E3EWXIwxxkScBRdjjDERZ8HFGGNMxFlwMcYYE3EWXIwxxkTc/wdmtQVnxeNF+AAAAABJRU5ErkJg
gg==
){:width="100%"}


Clearly there's some non-monotonic behavior here. This is interesting. I would
have thought that the power at $$x=0.5$$ would be a strictly decreasing function
of $$N$$--in other words, that there'd be less of a chance of drawing the
incorrect conclusion that the coin is biased, as we do perform more trials. This
is just based on the intuition that we usually have a higher probability of
drawing correct conclusions as the number of trials goes up.

Let's zoom in on the region from $$N=0$$ to $$N=200$$.


{% highlight python %}
N = np.arange(10, 210, 5)
z = [power(0.5, n) for n in N]
{% endhighlight %}


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$$x = 0.5$$')
plt.xlabel(r'$$N$$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
{% endhighlight %}




![png](data:image/png;base64,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
H1pKqc/lW10fAAAAAElFTkSuQmCC
){:width="100%"}


That's definitely interesting behavior. The overall trend of the curve seems to be going up, but there's some
periodicity too. You can see a series of
sawtooths, each of which is slightly higher than the previous one and whose
periodicity seems to be increasing. I wouldn't have predicted this. I'm not just referring
to the interesting shape of the graph, but the fact the the power is essentially
increasing with $$N$$ when $$x=0.5$$. This sounds good, until you realize that when
$$x=0.5$$ the power is the probability of drawing the _wrong_ conclusion.

#### Minimum sample size needed

We can also answer the question: given a p-value, and a suspected minimum bias
of the coin, what is the minimum value for the number of flips $$N$$ we need to
carry out so that the probability of concluding the coin is rigged is at least
0.8?

We can do this by fixing $$x$$ at the suspected bias value, and calculating the
power over a large range of $$N$$. Then we can just look for the first value of
$$N$$ for which the power is at least 0.8.


{% highlight python %}
x = 0.7
N = map(int, np.round(np.arange(4, 200, 1)))
powers = np.array([power(x, n) for n in N])
ind = np.where(powers >= 0.8)[0][0]
print('At least {0} flips'.format(N[ind]))
{% endhighlight %}

    At least 49 flips


If we suspect there's at least an 70% chance the coin is biased to land on heads
or tails, and we want to have an 80% chance of detecting this bias in an
experiment, we'd have to flip the coin at least 49 times in our experiment.

#### Testing

Another nice thing about examining a simple situation like coin flipping is that
we can do simulations to check our function. Since the output of our function is
a probability, we can use brute force to calculate it by running many trials,
using a random number generator, and seeing the proportion for which our
condition is satisfied.

In this case, since the power tells us the probability of making a conclusion in
an experiment, for every trial we have to simulate an experiment, where we flip
a coin $$N$$ times and use a $$p$$-value to conclude whether or not it's biased.
After a large number of trials, say 10,000, we look at the fraction of trials
where we concluded that the coin was biased. This fraction should equal the
theoretical probability our function calculates.


{% highlight python %}
from numpy.random import binomial
x = 0.3
N = 60
trials = 10000
print('simulated:   {0}'.format(sum(1 for i in xrange(trials) if pvalue(binomial(N, x), N) < 0.05) / float(trials)))
print('theoretical: {0}'.format(power(x, N)))
{% endhighlight %}

    simulated:   0.8352
    theoretical: 0.838182139392



{% highlight python %}
x = 0.5
N = 200
trials = 10000
print('simulated:   {0}'.format(sum(1 for i in xrange(trials) if pvalue(binomial(N, x), N) < 0.05) / float(trials)))
print('theoretical: {0}'.format(power(x, N)))
{% endhighlight %}

    simulated:   0.0405
    theoretical: 0.0400371916134


It seems our power function matches the brute force calculations. So we can be
confident that the analysis we did above is correct.

<!-- Does this imply that you have to think about adjusting $$p$$-values as you
increase $$N$$? Is that correct, wrong, deep, or obvious? I'm not sure which of
those apply. -->

---

[1] Coin flipping is like mouse models in biology. I heard of a cancer
researcher once saying something like "If you can't cure cancer in a mouse,
you'd better find another job". Similarly, if we can't apply statistical tools to a binomial model like
coin flipping, then we probably don't know what we're talking about.
