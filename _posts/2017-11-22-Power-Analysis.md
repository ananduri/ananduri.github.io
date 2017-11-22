---
layout: post
title: ""
category: posts
published: ""
---

# [{{ page.title }}]({{ page.url }})

# Power Analysis

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

## Inputs and outputs

Instead of medical experiments, let's confine ourselves to finding out whether a
coin is biased or not [1]. To fully specify the problem, let's state precisely
what we want to figure out. The question we're going to answer is,

>If a coin's bias is $x$, and we do an experiment recording $N$ coin flips,
what's the chance that we find statistically significant evidence that the coin
is biased?

If we answer this question, we'll be able to figure out how many trials of
flipping the coin we need--just keep increasing $N$ until we calculate that we
have a very high probability of detecting the bias.

We've also specified that we're looking for _statistically significant_ evidence
that the coin is biased. This means that we need to select a level of
significance. I'm going to use the typical value of 0.05--that is, if we observe
a $p$-value under 0.05 in our experiment, then we conclude that the coin is
biased. Technically, the significance level is another parameter you'd have to
supply when doing power analysis, but for the rest of the post I'm going to fix
it at 0.05.

So, an answer to the question would be a function or expression that you could
supply with the suspected bias of the coin, $x$, and the number of flips in your
experiment, $N$, and receive an output of a probability. Let's figure out that
function.

## Math

We still need to state what we mean by "a coin's bias is $x$". There's more than
one way to define this, but for this post $x$ will be the probability that the
coin lands on heads when you flip it. So an unbiased coin has $x=0.5$, but any
other value of $x$ means that a coin is biased.

We know what the probability of observing heads is when we flip the coin once.
What's the probability of observing $h$ heads when we flip the coin $N$ times?
This is the binomial distribution

\begin{equation}
P(h) = x^h \, (1 - x)^{N - h} \frac{N!}{(N - h)!\,h!}
\end{equation}

Let's write a function which we can supply with $h$ (and $x$ and $N$, of course)
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
coin is biased if the $p$-value is below 0.05. So we just need a way to
calculate the $p$-value from the outcome of the experiment.

To do that, assume that the coin is actually fair (this corresponds to our null
hypothesis). We need to calculate the probability of the outcome we observe _or
one more extreme_ occuring. In our context, say we observe 3 heads after 10 coin
flips. We need to find the probability of getting 3 heads if the coin was fair.
We also need to get the probabilities of observing all events more extreme than
3 heads, which means getting 2, 1, or 0 heads. The sum of these probabilities is
the $p$-value.

On the other hand, another way to phrase this outcome is: the difference between
the expected number of heads (5) and the observed number was 2. Put this way,
the more extreme outcomes correspond to an observed difference of 3, 4, and 5,
which means we'd have to also add up the probabilities for observing 8 heads, 9
heads, and 10 heads. Which way is right?

They both are. The first way of adding up probabilities corresponds to a one-
tailed $p$-value, and the second corresponds to a two-tailed $p$-value (since
we're adding up probabilities on both sides/tails of the distribution). If we
knew that the coin was biased to land on tails more often, then we'd use first
calculation, where we add up the probabilities for landing on fewer and fewer
heads. But we've stated only that we suspect the coin to be biased; we don't
know if it's biased towards landing on heads more, or on tails more (we're not
testing if $x > 0.5$ or $x < 0.5$; we're just making sure that $x\neq 0.5$. This
means that we add up the probabilities all outcomes where the difference from
$x$ and 0.5 is greater than what we observe, or the two-tailed $p$-value.


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
$p$-value that corresponds to this outcome?


{% highlight python %}
pvalue(4, 10)
{% endhighlight %}




    0.75390625



It's a very large number, and certainly not less than 0.05, which matches our
intuition that we have very little reason to suspect a bias. On the other hand,
what's the $p$-value for observing 1 head when flipping the coin 10 times?


{% highlight python %}
pvalue(1, 10)
{% endhighlight %}




    0.021484375



This $p$-value is 0.02, which is less than 0.05. That means that at this
significance level, we'd conclude that the coin is biased. This also matches our
intuition where we'd be suspicious if we only saw one head after 10 flips.

Now that we can calculate the $p$-value, and therefore know how to draw a
conclusion from every experiment, we can answer our original question: What is
the probability that we conclude the coin is biased? Let's write an expression
for that, including all our variables:

\begin{equation}
P(\mathrm{biased}, h, N, x)
\end{equation}

We're going to externally fix $N$ and $x$--in other words, they'll be parameters
we supply to the function that calculates $P$. Let's leave them out of the
expression. That means there's only one variable that we need to expand this in,
$h$:

\begin{equation}
P(\mathrm{biased}) = \sum_{H=1}^N P(\mathrm{biased} \mid h \;\mathrm{is}\; H) \;
P(h \;\mathrm{is}\; H)
\end{equation}

The binomial formula, encoded in the function Pheads we wrote above, gives us
$P(h \;\mathrm{is}\; H)$. To figure out the other factor, we know that we only
ever conclude the coin is biased if the $p$-value is less than 0.05. This means
that whenever $H$ is a value close enough to 0.5 such that the $p$-value is
above 0.05, there's 0 probability we conclude the coin is biased, and so
$P(\mathrm{biased} \mid h \;\mathrm{is}\; H) = 0$. Whenever $H$ is sufficiently
close to 0 or $N$ such that the $p$-value is under 0.05, the probability of
concluding the coin is biased is 1. Therefore the factor $P(\mathrm{biased} \mid
h \;\mathrm{is}\; H)$ has the effect of filtering out all terms in the sum for
which the $p$-value is not below 0.05.

Based on this intuition, the Python function that calculates
$P(\mathrm{biased})$, which we'll call the _power_, is:


{% highlight python %}
def power(x, N):
    return sum(Pheads(h, N, x) for h in xrange(N + 1) if pvalue(h, N) < 0.05)
{% endhighlight %}

## Giving it a spin

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
if the bias is as strong as a $3/4$ chance to land on one side of the coin.

## Plots with fixed N

Since we're now reasonably sure our code is correct, let's get a better sense of
the behavior of this function by making plots. We'll first look at the power
across all possible values of the bias, for different $N$.


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
    plt.plot(X, curve_dict[k], label=r'$N = {0}$'.format(k))
plt.legend(loc='lower left')
plt.xlabel(r'$x$', size=20, usetex=True)
plt.ylabel('power', size=16)
plt.axis([0, 1, 0, 1])
{% endhighlight %}




    [0, 1, 0, 1]




![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAY8AAAETCAYAAADOPorfAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAIABJREFUeJzsnXl8VOW5+L/vzGTfdyAhCdlIAglbWGRTVARR0eJSUOrW
alu1vV1u19tVa6vt7e1qW2vtT+sCrlUQBBdEUNawJZCFhCRkgexk3yaZ9/fHSSwiSyY558xkzvny
yWeSmXOe58lkOM9532cTUkpMTExMTEycweJqA0xMTExMxh6m8zAxMTExcRrTeZiYmJiYOI3pPExM
TExMnMZ0HiYmJiYmTmM6DxMTExMTp9HVeQgh/imEqBdCHL3A60II8UchRKkQIk8IMVNP+0xMTExM
hofeK49ngOUXef1aIHXw637grzrYZGJiYmLiJLo6DynlDqD5IofcCPxLKuwBQoUQ4/WxzsTExMRk
uNhcbcA5xAJVZ/1cPfjc6XMPFELcj7I6wRponeUV6aWLgUbEKqx4W73xtnjjbfXG3+aPv5c/FmGG
zEzUYUAO0GXvoqu/i76BPuwOO30DfQzIAVeb5tH0VPQ0SimjRnKuuzmPYSOl/Dvwd4C0jMny3kcf
52DlGSoau0iOCuD7y9OJCfHV2ooLPCVBOkDKwe8lOBwgB5QvRz/094GjT3m0d0JfF/R1QncLdDdB
VxO010JLJfS2/0d+cCyMnwYJl0HSleAfdmkrB1vQyMF/DulASuVxQA4wIAewO+zYB+z0DvTS099D
u72dtt422vraaOhqoLqjmpqOGmraa+iX/fTTz+TwySxNWMqq1FVE+kWq8o6aGIe6zjpeK3mN9yrf
o+RMCd5442/xJy4wjtigWOIC44jyiyLYJ5hg72CCvIPwtfoqNzJWb2wWGzZhwyIsWIQFIYTyOPgP
QAhxaUM6m6D0XajaC6fzoP2se1XfEAiNh6AY8IsA/wjwDQVvf/AOAC9/sPmAxQts3mCxgbCCxQrC
AkIAQnkUlsHvz2fEMOwcBdUt3Ty2uZDK5m6SowKYER/KzPgwluZccXKkMoXeva2EEInAW1LKqed5
7Ulgu5Ry3eDPxcAVUsrPrDzOJicnR+bm5gLwXkEd33z5MFaL4Pefn84Vk6PV/hX0RUroPgONx6E6
F6r3QdU+5QMurDBpMUxdBVm3gZfWzhK6+7vJb8jnQN0B9pzew8H6g9gsNpYmLOXOzDuZGvmZP6uJ
yac4VH+I5wqeY1vlNhzSwexxs5kzbg6zYmaRFZWFj9VHeyP6OiHvJTj6Opz8WLnZC46DiXOUr9hZ
EJEC/uHa26IxW47W8t+vHMHbZuFPa2awIOU/N3pCiANSypyRyHU353Ed8BCwApgL/FFKOedSMs92
HgAVjZ185fkDFNe18/OVU7jzskSVrHcTpITTR6DgDTj2Bpwph4BouOxByLkXfIN1M6WstYxXil/h
zdI36bB3sDZzLV+f8XV8bdo7MpOxRae9k9/m/pZXjr9CiE8Iq1JWcWvarUwMnqifEd1nYN9TsOev
0N0MkWmQeRNMuQmiMwdXCp7DX7ef4PEtRUybGMpf75jJhFC/T70+ZpyHEGIdcAUQCdQBPwW8AKSU
fxPKGvPPKBlZXcA9Usrc80v7D+c6D4DuvgG+tu4g24rqefqu2SxJH+MrkAshJVTshJ3/B2UfKMvs
y78Pc7+sLJ11otPeye8P/J71xetJDE7k0YWPkh2VrZt+E/dmf+1+fvzxjznVcYq7ptzFA9MfwM/m
d+kT1WLADrv+BDt/C30dkLoMFn0LJs71OIcxxFt5p3joxUPcMG0C/3trNj62z14Pxozz0IrzOQ+A
rr5+bv3bbk42dfH6A/NJiwlygXU6UnMQPngUSt+DuDlw458harKuJuw5vYeffPwT6rrqeGTBI6xM
XqmrfhP3Y33Reh7d+yjxQfH8YuEvmBE9Q18DTufBmw9CbR6kXw9X/ADGefb2al51C7f+bTdZsSG8
cN/c8zoOGJ3z8Oh0GX9vG0/dmYOft5UvPruf5s4+V5ukLbEz4Y5XYdVT0FQCf1sIH/9hMHCvD/PG
z+P1la8ze9xsfvzxj9lSvkU33Sbux79L/s2jex/lirgreOWGV/R1HA4HbH8MnlqiJJ/c9hysfsHj
HUdtaw/3/SuXyEAf/vaFWRd0HKPFo50HwIRQP566M4f6tl6++vwBBhxjf6V1UYSA7NvgwX2Qtgze
/Yly19Wvn+MM9A7kj0v+yPSo6fxg5w/YVrlNN90m7sPmss38dNdPmT9hPr+94rf4e/nrp9zeA6/e
A9t/BVNWwYN7IdPzV8F9/Q7ufy6Xjp5+/nFXDpGB2iUfeLzzAJg+MZRHP5fF3vJmnt1V4Wpz9CEw
WrnTuvz7cPgFePFW6GnVTb2/lz9PXPUEGREZ/PeH/82uU7t0023ierZVbuOHH/2QnHE5/H7J7/G2
euunvLMJ/rVSSShZ+gis+rtHZE0Nhyc/PEFedSu/vW0aGeO1TZwxhPMAuHlmLEsmR/GbrcVUNXe5
2hx9EAKW/ABu+itUfARPL4OOet3UB3oH8ter/0piSCI/2PkDzvSc0U23ieuo66zjRx/9iIzwDP50
5Z/0DYy3VMHTS+HUYbj1GVjwdY8NiJ9LaX07f9pWyvXZ41k+VfvGHIZxHkIIHv1cFhYBP/x3Pp6Q
KDBspt8Oa1+DlpPwwi2fLjrUmBCfEB5b9BhtfW38at+vdNNr4hqklDy852HsDju/XvxrArwC9FPe
1QzP3wydDXDXBpjyOf10u5gBh+S7r+bh72PlZyun6KLTMM4DlPjH969NZ2dJI68drHG1OfqSdAXc
+izUHoWX79Q1BpIWlsZXsr/C2+Vv8/7J93XTa6I/G8s2sqN6B9+Y9Q196zfs3bBujVLztPpFiJ+n
n2434LndFRysbOGnN2RqGuc4G0M5D4A75iYwOzGMR94qoKG919Xm6EvaNXDDH+DENtjwNV2zsO7N
upeM8Awe2fMILT0tuuk10Y/6rnoe2/cYM6NnsiZ9jX6KHQPw2peU9iKfexImLdJPtxtQfaaLX28t
5orJUdw0PVY3vYZzHhaL4Fersunq6+f/3j3uanP0Z+YXYMn/QN56+PBx3dR6Wbx4ZMEjtPa28tj+
x3TTa6IPUkoe2f0IfQN9PLzgYX2bZr77Eyh6C5b/SmnVYzAe31KMlPCLm6YOr5eXShjOeQCkRAey
dl4CL+2vpKROv/1/t2Hxd2DaGsV5nNQvC2py+GS+lP0lNpVt4mjjeeeBmYxR9tbuZXv1dh6a/hAJ
wQn6KS55F3b/GWbfB/O+qp9eN+FIVQsbj5zivkWTiAvTMRUagzoPgK9dmUqAt43HtxS52hT9EQJW
/AZCE+D1+5VOvjpx95S7CfYO5u95f9dNp4n2PJX3FFF+UazJ0HG7qqMe3vgqRE+Ba36hn143QUrJ
LzcXEhHgzf2XJ+uu37DOIzzAm68uSea9wnr2lDW52hz98QmCm59WuvO+9Q3d4h8BXgHckXEHH1R9
wPEzBtw29EAO1x9mX+0+7ppylz4dcUH5vL7xAPS0wc3/0KWjtLvxQXE9e8ub+cbVqQT66D9dw7DO
A+DeBZMYH+LLrzYXGit1d4i4WbDkh3Ds33D4Rd3U3pFxB/42f/6R/w/ddJpox1P5TxHqE8qtabfq
p3Tvk8oMjmWPQkymfnrdhP4BB7/aXMSkyABWz4l3iQ2Gdh6+Xla+tTSNI9WtbMq/6MgQz2XBNyBh
Ibz9XWjT5z0I8Qnh85M/z9aKrVS2Veqi00QbipqL2FG9g7UZa/VrP9JcrgTJ05bD7C/po9PNeO1g
NSX1HXxv+WS8rK65jBvaeQCsmhlH+rggfvvOcc/ve3U+LFal++6AXfkPqRN3TrkTm7Dx9NGnddNp
oj5P5T1FoFegvrGOrT8Eqxdc/3vDVI+fTW//AL9/r4QZ8aEsmzLOZXYY3nlYLYKvX5VKeWMnb+Wd
crU5riF8Esz/GuS/DCd366Iy0i+SVamr2HBiA7WdtbroNFGXstYy3j35LqvTVxPsrdMAspL3oHiz
kjEYrH0LDnfktQM1nG7t4ZtXp+mamnsuhnceAMunjCM1OpAnPijFYcTVByiDcYLj4O3vKEVXOnDv
1HuRUvJS8Uu66DNRl/VF6/GyeLE2Y60+Cvv7YMv3lPGw8x7QR6ebYR9w8JftpUybGMqi1MhLn6Ah
pvNAKRx86MoUjtd18E6BQe+CvQPgmkegNh8O/D9dVI4PHM9lEy5jU9kmHNKhi04TdbAP2Hm7/G2u
jL+SCL8IfZTu+Qs0lcLyx8GmY5deN+LNw6eoPtPN169McemqA0zn8QnXZY0nMcKfP20rNWbmFSiN
5BIXwbZfKE3mdGBl8kpOd57mQN0BXfSZqMNHNR/R0tvCDck36KOw7TTs+A2kXQupV+uj080YcEj+
8kEpmeODudINxmqbzmMQm9XCA0tSOHaqjQ+K9Wtb7lYIAdc+rsz9+Oh3uqhcMnEJAV4BbDixQRd9
JuqwsWwj4b7hzJ8wXx+FHz4O/b2w/Jf66HNDNuWfpqyxk6+5waoDTOfxKT43I5a4MD/++L6BVx8x
U2DqzbD/H9DZqLk6X5sv1yRcwzsV79Dd3625PpPR09rbyvaq7ayYtAKbRYfitJYqOPS80pctPEl7
fW6IwyH587YS0mICXZphdTam8zgLL6uFr1yezOGqFvaU6bNt45Ys/q7S4nrXH3VRd0PyDXT1d5nj
ascIWyu2YnfYWZms01jXj/5PeVz4LX30uSHbiuo5XtfBg0tSsFhcv+oA03l8hltmxRER4M0/dpa5
2hTXEZWmrD72/UMZ6akxs2JmMT5gPBvLNmquy2T0bDyxkZTQFNLD07VX1loNB5+DGWshVMf5IG7G
UzvLiA3147os90lPNp3HOfh6WVk7L4H3i+opre9wtTmu4/Lvgr0Ldv9Jc1UWYeH6pOvZfWo3DV0N
muszGTmVbZUcbjjMDck36LPvPhR7W2TcVUd+dSt7y5u5Z0EiNhdVk58P97HEjfjCZQl42yw8/VG5
q01xHVGTldkIe/+uy+rjhuQbcEgHm8s3a67LZORsLNuIQHDdpOu0V9ZaAwf/BTPugFDX9G9yB57a
WUaQj43Pz3avlZfpPM5DZKAPN8+M5fWD1TR1GGza4NksHlp9/FlzVZNCJpEVmcWmsk2a6zIZGVJK
NpdtZu74ucQExGiv8KPfgXQYOtZR09LNpvzTrJ4zkSBfL1eb8ylM53EBvrhwEr39Dp7fY+DGfdHp
kLkScp+Gvk7N1V0VfxWFzYXUddZprsvEecrbyqlsr+TqeB3qLLqalQyr7NUQpuNwKTfjmY+V3Y+7
F0xysSWfxXQeFyAlOoglk6N4bk8FPXZ92nW4JfMeUOo+jqzXXNXiuMUA7KzZqbkuE+fZWa38XRbF
6TAj/OC/oL8bLjNmGxKA9h476/dVsSJrPLGhfq425zOYzuMi3LcoicaOPt44VONqU1zHxLkwfroy
P0Hj2peU0BTGBYxjR/UOTfWYjIwd1TtICU1hQuAEbRUN9Ct1RomLlLojg/LS/irae/u5b5H7rTrA
dB4X5bLkCDLGB/PMrgrjFg0KAXO/Ao3FUPaBxqoEi2MXs+f0HvoG+jTVZeIc7X3tHKw7qM+qo3gT
tFYpnzuDMuCQ/Gv3SWYnhpEdF+pqc86L6TwughCCuy5LoKi2nX3lBi4anLoKAqKU1YfGLI5bTHd/
N7m1uZrrMhk+u0/tpl/2szh2sfbK9j6pZFdNvlZ7XW7K9uJ6Kpu7uGt+oqtNuSCm87gEN06PJcTP
i2d3V7jaFNdh84Gce+H4Vmg6oamqOePn4GP1YUeNuXXlTuyo3kGQdxDTo6drq+h0Hpz8GObcrwwq
MyjP7KogJtjHbVqRnA/TeVwCP28rn589ka3H6jjdauDeSzn3Kv+Z9z2lqRo/mx+zx81mR/UO424V
uhkO6WBnzU4WTFigfS+rvU+Cl79SUW5QTjR0sLOkkTvmJrhsxOxwcF/L3IgvzEvAISUvGDltN2ic
0rL90PPQ266pqsVxi6lqr6KirUJTPSbDo6CpgOae5k+y4TSjswnyX4Fpq8EvTFtdbsxzu0/ibbWw
Zo57F0aazmMYTAz356r0GNbtq6S338Bpu7Pvg752OPaGpmqGLlJm1pV7sKN6BwLBgtgF2irKewkG
epXPmUHp6O3n1QPVXJc9nqggH1ebc1FM5zFM7p6fSFNnH5vyTrvaFNcxcQ5EpimrDw2JDYwlOST5
k7oCE9eyo3oHWVFZhPuGa6dESuVzFTsLYjK10+PmvH6wmo7efrcOlA+hu/MQQiwXQhQLIUqFEN8/
z+vxQogPhBCHhBB5QogVett4PhakRJAcFcCzuypcbYrrEELZi67aA40lmqpaHLeYA3UH6OgzcHNK
N6Cxu5FjTce0z7I6dQjqj8H0O7TV48ZIKXl2VwXT4kKYPtE903PPRlfnIYSwAk8A1wKZwBohxLm3
GT8CXpZSzgBWA3/R08YLIYTgzssSOVLdSn51q6vNcR3Zq0FYNV99LIxdSL/sN8fTupi9p/cCsDBu
obaKDj0PNl9lFIBB2V3WxImGTr5wWaKrTRkWeq885gClUsoyKWUfsB648ZxjJBA8+H0IcEpH+y7K
52bG4udl5YW9J11tiusIioG0ZXBknVIJrBHZUdl4WbzIrTPrPVxJbl0ugV6BpIdpOLvD3g35r0LG
SvBz/zturXhhbyUhfl5cn+0+Mzsuht7OIxaoOuvn6sHnzuZnwFohRDWwGfja+QQJIe4XQuQKIXIb
GvSZARHs68WN0yfw5uFTtPXYddHplsxYCx11cOJ9zVT42nzJiswyiwVdTG5tLjOiZ2DVsuaiaBP0
tho6Pbe+vYetR2u5dVYcvl5jo77FHQPma4BnpJRxwArgOSHEZ+yUUv5dSpkjpcyJiorSzbg75ibQ
bR/g9QPVuul0O1KvUSrODz2nqZpZMbMobC6k0659R1+Tz9LY3UhFWwU543K0VXToOaWiPFGH1idu
ysv7q+h3SG6f697puWejt/OoAc6eaBI3+NzZfBF4GUBKuRvwBSJ1sW4YZMWFMC0uhOf3Vhq3iM3q
peTiF78NnY2aqckZl8OAHOBw/WHNdJhcmKF4U06Mhs7jzEko+xCmrwWLO97Las+AQ7JuXxULUiJI
igp0tTnDRu+/1n4gVQgxSQjhjRIQ33DOMZXAVQBCiAwU5+FWs0nvmJdAaX2HsftdTV8Ljn4lN18r
FVHTsQmbGfdwEbm1ufjZ/MiIyNBOyVCr/+lrtNPh5mwvrqempZu1c8fW3BJdnYeUsh94CNgKFKJk
VR0TQjwshFg5eNi3gfuEEEeAdcDd0s1u8W/InkCwr43n9xq44jw6XWnVnv+qZir8vfzJjMw04x4u
IrdOiXd4WTSaYCelUlGeuNDQY2af33OS6CAfrs7UYTqjiui+TpRSbpZSpkkpk6WUjw4+9xMp5YbB
7wuklAuklNOklNOllO/obeOl8PO2cvOsOLYcPU1Du4HH1GbdAqcOatosMScmh6NNR+nuN3BfMRdw
pucMpS2l2m5Z1eZBU4mh03OrmrvYfryB1bMnunUfq/Mxtqx1I+6Ym4B9QPKqkQPnU1Ypj0df10zF
rJhZ9Dv6OdJwRDMdJp/lYN1BQHn/NSP/VbDYIPPcbH3jsH5/JQJY7eZ9rM6H6TxGSEp0IHMmhbN+
fyUOh1vtqulHSCzEz4ejr2o2ZXBm9EwswmIWC+pMbl0uPlYfpkZO1UaBw6HcdCRfCf4atj1xY+wD
Dl7OrWbJ5GgmuOGY2UthOo9RcPuceE42dbG7rMnVpriOrJuhoQjqjmkiPtA7kPTwdDPuoTMH6g4w
LWoa3lZvbRRU7YW2aph6izbyxwDvF9bT0N7r9t1zL4TpPEbB8qnjCPX34kUjB84zb1LalRzVLnCe
E5NDXkMevQMGji/pSFtfG0XNRdrGO46+qrQjSXeL1nUu4cV9lYwP8eWKyfrVqamJ6TxGga+XlZtn
xrH1WK1xA+cBkZC8BI6+ptnWVU5MDn2OPvIb8jWRb/JpDtUdQiK1Kw4c6Ffa+qctB58gbXS4OVXN
XewsaeC2nInYxligfIixabUbsWbORPodktcOGjhwPvVmaKmEam22lmbGzEQgzHoPnThQdwAvixdZ
kVnaKCj/ELoaDZ1l9dL+KgTw+dkTL3msu2I6j1GSEh3EnEnhrNtn4MB5+vVg9dFs6yrEJ4Tk0GQz
40onDjccJjMiE1+brzYKjr4GPsFKmxsDYh9w8FJu1ZgNlA9hOg8VMHzg3DcY0q6BY/9Wsmg0IDsq
m6ONR43bEkYn7A47BU0FZEdla6Ogvw8K34L068BLI+fk5oz1QPkQpvNQATNwjhI476iD6n2aiM+K
zKKlt4Wq9qpLH2wyYkrOlNA70Et2pEbOo/xDpYNu5k3ayB8DrNtXybjgsRsoH8J0Hirg62Vl1Yw4
3imopanDoIHz1GvA6g2FGzURP7T/nteYp4l8E4WhpISsKI3iHYUbwDsQkq7QRr6bU9XcxY6SBm6b
PXYD5UOMbevdiDVzJmIfMHDg3DdYuSAUbtAk6yolNAU/mx95Dabz0JK8xjzCfcOZEDBBfeGOAWV2
R9oyw25ZvZKrrJxvy4lzsSWjx3QeKpEaE0ROQhjr91UZd18+Y6WSdVWr/gXearEyJWKKma6rMXkN
eWRHZiOEUF945W7oaoKMG9SXPQboH6wovzwtirgwf1ebM2pM56Eiq+fEU9bYyV6jtmqfvAKERbOt
q+yobIrOFJnFghrR2ttKRVuFdsHywo1KVl7KUm3kuznbixuobeth9eyxHSgfwnQeKnJd1niCfG2s
32fQwHlABCQs0M55RGbT7+inqLlIE/lG51ij0mJGk3iHlMrnIuUq8Bk7A4/UZP3+SiIDfbgqI9rV
pqiC6TxUxM/byudmxLL5aC0tXX2uNsc1ZKxUel01HFdd9NBFzdy60oa8xjwEgqkRGjRDrDkIbTWG
3bKqbe1hW1E9t+XEjbnW6xfCM34LN2L17Hj6+h28fvDc6boGIf065bHw3AGRoyfaP5oY/xgzaK4R
eQ15JIUkEeitwcqgcIPSfj1tufqyxwAv51bhkGO7ovxcTOehMpkTgpkWF8L6/QadcR4SC7E5msY9
zHRd9ZFSkt+Yr+GW1QZIXGTI9usOh+Sl/cqM8oSIAFeboxqm89CA1XPiOV7XwcHKFleb4hoyboDT
h+HMSdVFZ0VmUdNRQ3OPQZMSNKK6vZqW3hZt+lnVF0JzmWG3rHaWNlLT0u0xgfIhTOehATdMm0CA
t5V1Rg2cD10kit9WXfRQJpAZ91CXodXctKhp6gsv2qQ8Dm1pGox1eysJD/Dmmilja0b5pTCdhwYE
+thYOX0Cb+Wdoq3H7mpz9CciGSLT4Lj6ziMzIhOrsJpbVyqT35iPn82P5NBk9YUffxtiZ0HQOPVl
uzkN7b28V1jHzTNj8bFZXW2OqpjOQyNWz46nx+7gzcOnXG2Ka5h8LVR8BD2tqor1s/mRGpZqBs1V
Jq8hj8yITGwWm7qC22uh5oDyeTAgrx6opt8h+byHbVmB6Tw0IzsuhMzxwazba9DAedq14OiH0vdU
F50VmcWxxmM4pDYdfI1G30AfRc1F2jRDPL5FeUwznvOQUvLS/krmTAonJdrzaltM56ERQgjWzJlI
wek28mvUvfseE0ycA37hULxFddFTIqbQbm+nut2gfcRUprSlFLvDTmZkpvrCi7dASDzETFFftpuz
u6yJiqYu1szxnPTcszGdh4bcOCMWXy8L6/YZsI24xark9Je8o4wdVZHMCOUiV9BUoKpcozL0Pk4J
V/kC39cFZR8oW1Za9Mpyc9btqyLY18a1U8e72hRNMJ2HhgT7enF99gQ2HK6hs1fdC+iYYPJy6GmB
qj2qik0JTcHL4kVBs+k81KCwqZAgryDiglTu9Fr+IfT3KJ8Dg9Hc2cfWo7WsmhmHr5dnBcqHMJ2H
xqyZM5HOvgE2HDFg4Dz5SmXGh8opu15WL1LDUs2Vh0oUNBWQEZGhfifd4s3gHQQJC9WVOwZ4/WA1
fQOOMT8t8GKYzkNjZsaHkRYTaMxmiT5BMGmxchFROWkgMyKTwqZCYyYjqIjdYef4meOfbAWqhsOh
xDtSrwabt7qy3RwpJev2VTIzPpTJ44JcbY5mmM5DY5TAeTxHqls5asTA+eRrlerixhJVxWaEZ9DW
10ZNh0F7iKlEWUsZfY4+MsIz1BV86hB01itt+g3G/ooznGjo9OhVB5jOQxdWzYjDx2Zh/X4Drj6G
GuEVb1ZV7JQIJbhrbl2NjqH3T/WVR/FmEFZIuVpduWOAdfsqCfK1cX22BtMY3QjTeehAiL8X12WP
541Dp+jqM1jgPCQOxmX9J99fJVLCUrAJG4XNharKNRoFTQUEeAUQH6zyXfLxLRA/z3CNEM909rEp
/zSrZsTi5+2ZgfIhTOehE7fPiaejt5+3jpx2tSn6k7oMqvZC9xnVRPpYfUgJSzFXHqOkoLmA9PB0
LELFS0FrDdQdhdRr1JM5Rnj9UA19/Q7WzPXsLSswnYduzEoIIzU6kBeNGDhPWwbSAaXvqyo2MyKT
gqYCM2g+Qvod/Rxv1iBYXvKO8pi2TF25bs5QoHxGfCjp44JdbY7mmM5DJ4YC54erWig41eZqc/Ql
dhb4R/znoqISGeEZtPS2UNtZq6pco1DeWk7PQI/6wfKSd5Sq8qh0deW6Obknz1Ba3+HxgfIhTOeh
I6tmxuJtsxivVbtlMHBa8i44BlQTa1aaj45PKssjVKwst/dA2XZIu8ZwVeUv7q0kyMfG9dmeWVF+
Lro7DyHEciFEsRCiVAjx/Qscc5sQokAIcUwI8aLeNmpFqL8312WN541DNcYLnKdeA93NSodVlUgL
S8MqrGal+QgpbC7Ez+ZHQnCCekJPfgT2LiXOZSCGAuU3zYjF31vlzsRuiq7OQwhhBZ4ArgUygTVC
iMxzjkkFfgAskFJOAb6hp41ac8fceNp7+9lotIrzlKuU1E0Vt658bb4khSaZK48RUtCkBMutFhWz
gkreBZsemj6KAAAgAElEQVQvJBqrqvy1g9X09Tu43QCB8iH0XnnMAUqllGVSyj5gPXDjOcfcBzwh
pTwDIKWs19lGTZmVEMbkmCBe2GuwrSu/MJg4F45vVVVsZrgZNB8JA44BipqL1A2WS6n8fSctBm9/
9eS6OVJKXtxbyayEMDLGe36gfAi9nUcscHaL2erB584mDUgTQnwshNgjhDhvVzUhxP1CiFwhRG5D
Q4NG5qqPEII75sWTV91KXrXBZpynXQO1edCmXrpyRkQGzT3N1Hd51D2G5pxsO0l3f7e6wfKmUjhT
brgU3d1lTZQ1dnKHgVYd4ITzEEJ4CyH+LYRYrKVBgA1IBa4A1gBPCSFCzz1ISvl3KWWOlDInKipK
Y5PU5aYZsfh5WXnRaKuPoX1wFbeuzErzkXGs6RigOF/VGFpVGixF94W9lYT4ebEiyxiB8iGG7TwG
t5muduac81ADnD0ZJW7wubOpBjZIKe1SynLgOIoz8RiCfb1YOW0CG44YbMZ5dAYEx6nqPNLC0hAI
is4UqSbTCBQ3F+Nj9SEpJEk9oSVbISoDQo1zB97Q3ss7x2q5ZZbntl6/EM46go+BeaPQtx9IFUJM
EkJ4A6uBDecc8wbKqgMhRCTKNlbZKHS6JXfMi6erb4A3DxmosZ8QytbViQ+gv1cVkf5e/iQEJ1DU
ZDoPZyhqLiI1NFW9meW97XByN6QuVUfeGOGVA1XYB6ShAuVDOOs8vg18UQjxkBAiTghhFUJYzv66
2MlSyn7gIWArUAi8LKU8JoR4WAixcvCwrUCTEKIA+AD4jpSyyUk73Z7suFCyYkN4wWgzzlOXgb0T
KnerJjI9PJ2iZtN5DBcpJYXNhUwOn6ye0LLt4LAbKt7hcCiB8suSIkiO8rwZ5ZfCWeeRDyQDfwBO
An2A/ayvvksJkFJullKmSSmTpZSPDj73EynlhsHvpZTyW1LKTClllpRyvZM2jhlunxtPUW07B06q
1/PJ7Zm0SBkQVfKuaiLTw9M51XmK1l4DtrwfAbWdtbT1takbLC95B3yClWaIBuHDkgaqz3QbctUB
SnDaGR4GDHSbrC03Tp/ALzcV8tyek+QkGqT7qHeAUgNQ8i4se1QVkenhShuM4uZi5oyfo4pMT2ao
E7FqKw8poeQ9SLoCrF7qyBwDPL/7JJGBPiybMs7VprgEp5yHlPJnGtlhSPy9bdw8K44X9p7kx9dn
Ehno42qT9CFlKWz9AZw5CWGjr24euggWNReZzmMYFDcXIxCkhaWpI7DuGLSfMlS8o6q5i23F9Ty0
JAVvmzG7PI34txZCBAohEoQQxrnV0IC18xKwD0he2l916YM9haF98VJ1tq4i/SKJ8osy4x7DpLC5
kITgBPy9VCrkG/o7phjHeby4rxIBhmmCeD6cdh5CiOuFEAeBVpQsqKzB5/8hhLhdZfs8npToQOYn
R/Di3koGHAbZEYxIhrBEZatDJdLD08103WFS3Fz8yVafKpS8pwz8CjZGnUNv/wAv7a/i6owYJoT6
udocl+GU8xBC3AS8CTQC3wPObptZDtylnmnG4QvzEqhp6eaDIoNUSQuh3KWWf6h0YVWB9PB0ylrK
6B1QJwXYU2ntbeVU5yn1nEdPq5I5Z6BVx9v5tTR39vGFy1RsKDkGcXbl8VPg/0kprwF+f85rR4Gp
qlhlMJZmxhAT7MNze0662hT9SL1G6b5auUsVcenh6QzIAUpbSlWR56kUNxcDqOc8yraDHDBUiu5z
e06SFBnAguRIV5viUpx1HhnAS4Pfn7vHcgaIGLVFBsRmtXD7nAQ+PN5ARWOnq83Rh8SFYPVRLWV3
6GJoFgteHNUzrUreAd8QiJutjjw359ipVg6cPMMd8xKwWIw1r+RcnHUebcCF3G0iMHY6FLoZa+ZM
xGYRPG+U1Ye3v1LzoZLziAuKI8ArwAyaX4Li5mKi/KKI9FPhrnkoRTf5SrAaY4bFc7tP4utl4ZaZ
ca42xeU46zzeBX5wTqNCKYTwQakcf1s1ywxGdLAvy6aO4+XcKuMMikpZCk0l0Fw+alEWYWFy2GTT
eVyCwuZC9basavOho9Yw8Y6Wrj7eOFzDTdNjCfE3k0yddR7/A4wDioF/oGxdfR84jNLk8GdqGmc0
7pmfSFtPP/82Sr+roboAFbeuis8U45AOVeR5Gr0DvZS3lqvnPD5J0b1aHXluzkv7q+ixO7hrfqKr
TXELnHIeUsoKYCbwFrAUGAAWA3uAuVJKg43HU5dZCWFMmRDMs7sqjNHvKiIZwpNUq/dID0+nu7+b
yjaDtbofJqVnShmQA+o5j5J3Yfw0CIpRR54bM+CQ/Gv3SeZOCjfUwKeL4XSdh5SyWkr5RSllnJTS
W0o5Xkp5j5TSQFVu2iCE4K75iRyv62D3CY/rBXl+Uq+B8p1g7x61qE+C5ma9x3kZ2tJTxXl0n4Gq
fYbJsnqvsI6alm7uNlcdn+Bsnce1QogArYwxgZXTJhAe4M0zuypcbYo+pCyF/m6o+Hj0okJTsFls
ZsbVBShsLiTAK4C4IBWCvSc+UFJ0DRLveHZXBRNCfFma6fmrrOHi7MpjE9AshNglhHhUCHGVEMJX
C8OMiq+XldWzJ/JeYR3VZ7pcbY72JC4Am58qW1deVi+SQ5LNoPkFKG4uZnLYZCwXn5wwPErfU+bS
x+WMXpabc7yunV0nmlh7WQI2qzH7WJ0PZ9+JNODrKO3Yv4iSfXVGCPGhEOKnOoyoNQRr5yUghDBG
0aCX32DKrjrTBc3ZHudnwDFA8RmV2pI4HEq8I/lKsHj+9Lxnd1XgbbOwerZx+1idD2cD5qVSyiel
lGuklONQKsq/A/QDPwG2aWCj4ZgQ6seyKTGs32eQtN3Ua6C5DJpOjFpUeng6TT1NNHY3qmCY51DV
XkV3f7c6zqP2CHTWGyLe0dpl5/WDNdw4uJ1s8h9GtAYTQvgLIZYBd6L0s7ocpYDwLRVtMzT3LJhE
a7ed1w4aIG13KNVThZTdoYtjYVPhqGV5EqoGy4caWiZfNXpZbs6L+yrptg9wz4JJrjbF7XA2YP6w
EOIjlFYkrwLTgJeBuUCElPIm9U00JjkJYWTHhfD/PirH4enddsMnQUSqKltXQ203is8Uj1qWJ1HU
XIRN2EgOTR69sJJ3YMJMCIwavSw3xj7g4NldFcxPjiBzgpmeey7Orjx+BEwH/ggkSSmvlVL+Rkp5
QEqzMktNhBB8ceEkyho72X7cAN12U5dCxUfQN7okgSDvIOIC48yVxzkUNReRHJqMt3WUWy9dzVCT
a4jBT5vzT1Pb1sMXF5qrjvPhrPP4L+Ad4F7gtBDigBDiN4MpvMabAK8xK7LGMy7Yl6c/Gn37Drcn
dSkM9ELFzlGLMoPmn6WouUidZogntoF0eHy8Q0rJPz8qJykygCWTo11tjlvibMD8T1LKVSjNEecA
L6B02l2HksI7+mR9k0/wslq4c34CH5c2UXi6zdXmaEvCAvDyVy3uUdleSafdIB2KL0FDVwNNPU1k
hGeMXljJu+AXDhNmjF6WG3Pg5BmOVLdyz4JEw3fPvRAjCphLpXfGUeAgcAgoQpmHPk8900wAbp8T
j5+XlX96+urD5gOTLoeSrUq31lEwFBQeml1hdIZWYaNeeTgcSn1HylUen6L79EflhPh5cfMss3vu
hXA2YD5fCPEjIcT7QAvwPvBloBJ4EJiivonGJtTfm5tnxfLm4VM0tHv4lLzUpdBSCY3HRyXmkzYl
5tYVoGKm1amD0NUIqctUsMp9qWruYuuxWtbMicff2xit5keCsyuPj4BvAe0o3XSnSymjpZS3SSn/
KqU0/7dqwL0LJmF3OHhud4WrTdGWoX3041tHJSbaP5ownzDTeQxS1FxEbGAsQd5BoxN0fCsIi7Ly
8GD++XE5FiG4a76xx8xeCmedRw6DKblSyj9KKfO1MMrk0yRFBbI0I4Z/7Tnp2UWDoRMhesqoU3aF
EGbQ/CyKmotUindshbg54B8+elluSktXH+v3VbFy+gTGh/i52hy3xtmA+cHBeAdCiEAhxEQzy0of
vnx5Ei1ddl7e7+HNi1OXQuVu6GkdlZj08HRKW0qxO+wqGTY26bR3UtleOfp4R3stnD7i8Sm6z+85
Sbd9gPsXJ7naFLfH6YC5EGKZECIXJeZRAbQIIfYJITz7U+ViZiWEk5MQxj8+Kqd/wINLatKWgaNf
6do6CtLD07E77JS1lKlk2NhkKGlg1CuPoSy4NM+Nd/TYB3hmVwVXTI4ifZxZFHgpnA2YL0PprBsI
PAI8APwCCAI2mw5EW+5fnET1mW42H611tSnaETcHfENHvXVlBs0VVMu0KtkKwbEQM1UFq9yT1w/W
0NjRZ646homzK4+foRQJZkopfz7YJPFnKFlW7wI/V9c8k7O5OiOGpKgA/r7jhOdOGrTalIBsyTtK
augISQhOwM/mZ3jnUXymmDCfMGL8RzGHor8PTmxXtqyEZ9Y8DDgk/9hZRnZcCJclRbjanDGBs85j
GvDEua1IBn/+C0rrEhONsFgE9y9K4mhNG7s8edJg6jLobIDTh0YswmqxkhqWanjnUdhUSHp4OmI0
F/3KXdDX7tEpuu8W1FHW2Mn9i5NG914ZCGedRy9woc3AoMHXTTTkphmxRAX58LcPR9++3G1JuQoQ
o642Tw9Lp7i52HNXaZfA7rBT2lI6+vqOknfB6g2TPHNcj5SSJ3ecYGK4H8unjHO1OWMGZ53HduAR
IcSnOoUJIeJRtrRGF+U0uSS+Xla+uHASO0saOVLV4mpztCEgUplQN8p6j/SIdNrt7VR3VKtk2Nii
rKUMu8M++njH8a2QuBB8PDOxcveJJg5VtvDlxcnmpEAncPad+h4QAhQLIXYIIV4SQnwIlAChg6+b
aMzaeQkE+9p44oNSV5uiHanLlIrmjpF3FM4MzwSMO9ujoKkAgMyIzJELaS6DphKP3rL68welRAf5
cIvZisQpnK3zOA5ko7Rk9wFmAr7AH1CqzUtUt9DkMwT62Lh7wSTeKaijuLbd1eZow1BK6ChWH6lh
qdiE7ZOLqNEoaCrA3+ZPQvAoKqWLtyiPaZ7ZRfdg5Rl2nWjivkVJ+Hp5dr8utXF6jSalPA08DHwT
+OHg4yODz5voxD3zE/H3tvLX7R66+hiXBcFxcHzLiEV4W71JCUuhsNmYK4/CZiVYbhGj2Io5/jZE
pUO4Z6av/uWDUkL9vbh9rjmf3FlGUiT4E6AK2AmsH3ysFkL8aJjnLxdCFAshSoUQ37/IcTcLIaQQ
IsdZG41AWIA3a+clsOHIKU42eWDrcSFg8nJlfoS9Z8RiMiMyKWgqMFzQvN/RT3Fz8ei2rLpb4OQu
mHyteoa5EYWn23ivsJ57F0wiwMdsgOgszhYJ/hwlMP4SsBRlC2spyijanwshfnaJ863AE8C1QCaw
RgjxmU+3ECIIZfDUXmfsMxpfWjgJm9XiuZlXadeCvQvKd4xYREZ4Bi29LdR2enBh5Xkoby2nZ6Bn
dM6j9D2l2j/NM53HEx+UEuhj467LEl1typjE2ZXHfcBvpZT3Sym3SSmPDT7eB/wOuP8S588BSqWU
ZVLKPpSVy43nOe4R4HFg5LecBiA62JfbcuJ49UA1p1q6XW2O+kxaBN6BytbJCBm6eBot7jG0VTcq
53F8C/hHKJlvHsaJhg42559m7bwEQvy9XG3OmMRZ5xECXCiCuWXw9YsRi7LlNUT14HOfIISYCUyU
Um66mCAhxP1CiFwhRG5DQ8Ml1HouX7k8GYC/eGLsw+YDyUuUoO0It53SwtKwCivHmo6pbJx7U9BU
gJ/Nj8TgxJEJGLArVf5pyz1y8NOf3i/Bx2blS4vM+eQjxVnnsReYfYHXZjPKbSYhhAX4P+DblzpW
Svl3KWWOlDInKipqNGrHNHFh/tyaM5GX9ldR44mrj8kroP2U0tF1BPjafEkKTTJc0LygqYD08HSs
I73wV+5ROhunLVfXMDegtL6DDUdOcedlCUQG+rjanDGLs87j68C9QojvCCEShRB+g4/fBe4FHhJC
WIa+znN+DTDxrJ/jBp8bIgiYCmwXQlSgjLXdYAbNL86DS1IAJXPE40i9BhBQPIqtq3BjBc0HHAOj
n+FR/LZSVZ58pXqGuQl/2qasOswGiKPDWeeRByQDjwEngI7Bx18NPp8P2Ae/+s5z/n4gVQgxSQjh
DawGNgy9KKVslVJGSikTpZSJwB5gpZQy10k7DUVsqB+35Uzk5dwqqs90udocdQmIhIlzRxX3yIjI
oLmnmfqukRccjiVOtp2ku7975PEOKaF4s9KOxMOqykvr25VVx/wEIsxVx6hwNj/tYWDEt29Syn4h
xEMocRMr8E8p5TEhxMNArpRyw8UlmFyIB5ek8HJuFX/ZfoJffi7L1eaoy+Rr4b2fQmsNhMRe+vhz
mBIxBVC2cmICRtFddowwFN/JiBjhyqPxOJwph/kPqWiVe/DH90vx87Jy/yJz1TFanHIeg+3XR4WU
cjOw+ZznfnKBY68YrT6jMCHUj8/PVmIfD1yRTFyYv6tNUo8h53H8bZj9JadPTwtLwyIsFDYXsiR+
iQYGuheFzYX4Wn1JChnhBXJoi9DD4h0lde1szDvFlxcnm6sOFTC7gHkQD1yRgkDwx/c9rEtMZJpS
4Vy0+dLHngd/L38mBU8yTLpuQVMBaeFp2CwjLHwr2gTjsiHEs3o9/e6948qqw4x1qILpPDyICaF+
3DEvnlcPVFNa3+Fqc9RDCMi4QSkW7B5ZJ+GhSnNPxyEdowuWt9dC9T7IWKmuYS4mv7qVzfm1fGnh
JMIDvF1tjkdgOg8P48ElKfh5Wfm/d4tdbYq6pN8ADvuIx9NmRGTQ0N1AQ5dn1wRVtlXSae/8JM7j
NEVvKY8Z16tnlBvw661FhPl78SVz1aEapvPwMCIDffjioiQ259eSV+1B8z5iZ0HQeCgcWU7FUOaR
p9d7jLoNe+FGiEhRmiF6CLtPNLGzpJEHrkgh2NesJlcL03l4IPctmkSYvxe/2epBqw+LBdKvg9L3
oc/5dOT08HQEgmONnl1pXtBUgLfFm6TQEdxhdzVDxUfKFqGHjGKVUvLrrUWMD/HlC5eNojW9yWcw
nYcHEuTrxYNLUthZ0siu0kZXm6MeGTcojRJPbHP61ACvAJJCkshvzNfAMPchvzGf9Ih0vCwjuMM+
vlVphJh+g/qGuYh3C+o4VNnCf12Vas7rUBnTeXgoa+clMD7El8e3etAM74QF4BuqbK2MgKyoLPIb
8z3n/TgHu8NOQVMB2ZHZIxNQuBGCY2HCDHUNcxEDDsn/vlNMUmSAOSVQA0zn4aH4eln55tI0jlS1
sDHPQ+Z0Wb2UXlfH31Ya9zlJVmQWLb0tVLd75kzz0jOl9Az0kBU5giLRvk448T6kX69sEXoAL+dW
cbyug+8sm2zOJtcA8x31YG6eGUfG+GAef7uIHvuAq81Rh4zrlYZ9FTudPjU7Srkjz2vMU9sqt2Bo
Sy4ragTOo/Q96O/xmCyrjt5+fvtOMbMTw1g+dZyrzfFITOfhwVgtgh9dl0FNSzfP7KpwtTnqkHwl
ePlD4VtOn5oSmoKfzc9j4x55DXmE+YQRFziCLZrCt8AvHOLnq2+YC/jb9hM0dvTxP9dlIjwk+O9u
mM7Dw1mQEsmV6dE8sa2Upo5eV5szerz8IOVqpR7B4dxqymaxkRGeQX6DZzqP/MZ8sqKynL9Y9vcq
wfLJK8A69sexnmrp5qmdZaycNoHpE0NdbY7HYjoPA/DDFel02Qf4g6e0LZlyE3TUQeVup0/Njsqm
sLmQvoHzNX0eu7T3tVPeWj6yeMeJbdDbqryvHsD/bi1GAt9dPtnVpng0pvMwACnRQayZM5EX9lZS
Wt/uanNGT9pyZevq6GtOn5oVmYXdYae42YNqYICjjUeRyJFlWh19DfzCIOkKtc3SnbzqFl4/VMO9
CyZ5VnNQN8R0Hgbhm1enEeBt5WcbPGAokneA4kAK3oSBfqdO9dSg+VAcZ2rUVOdO7OtSuuhmrFSy
2cYwDofkJ28eIzLQhweWJLvaHI/HdB4GISLQh28tTeOj0ka2Hqt1tTmjZ+oq6GqC8g+dOi3GP4Yo
vyiPC5rnN+aTGJxIsHewcyeWvAN9Hcr7OcZ57WA1h6ta+P616WYbEh0wnYeBWDsvgfRxQTzyViHd
fWM8dTdlKXgHwbHXnTpNCEFWZJZHBc2llOQ35H+yqnKKY69DQBQkLFTfMB1p67Hz+JYiZsaHsmqG
8wPDTJzHdB4Gwma18LOVU6hp6eavH55wtTmjw8tX6XVVuBH6nQt+Z0VlUdleSUuPZzSOPN15mqae
JueD5b3tcPwdyLxpzGdZ/f7dEpo6+3j4xqlYLGZqrh6YzsNgzEuKYOW0CfztwxNUNo3xeedTb1YK
Bp3sdTUUVPaUrauh+I3TxYHFW6C/e8xvWRXXtvPs7grWzIlnamyIq80xDKbzMCA/XJGBzSL42cZj
Yzt4nnSF0uvKya2rKZFTEAiPcR75Dfn4WH1IC0tz7sRjr0PQBJg4TxvDdMDhkPz4zaME+tj4zjVm
aq6emM7DgIwL8eVbS9PYVlTP5vwxHDy3eUPmSmU8rb172KcFeAWQHJrsMRlX+Y35ZIRnONdJt7tF
aUky5XNjupfVKweq2FfezA+uTSfMnBCoK2P3U2MyKu6en8jU2GB+tvEYrd3ONxl0G6asgr52pULa
CaZFTSOvIY8BJ6vU3Y2+gT6lk66zwfKit2Cgb0xvWTW09/LopkLmTArntpyJrjbHcJjOw6DYrBYe
W5VNU0cvj28pcrU5I2fSYmXC4JH1Tp02K2YW7X3tlLaUamSYPhxtPErvQC+zYmY5d+LhdRCerExo
HKM88lYBPXYHv/xclhkkdwFjO8XiItjtdqqrq+np6XG1KW6LFXjhlokcretkf1kDs5OiXG2S81is
kH0b7H4COhogcHi/w9DFNrcul8nhY3evPLcuF4CZ0TOHf9KZCjj5ESz50ZidGLi9uJ4NR07xX1el
khId6GpzDInHOo/q6mqCgoJITEw0u2pehP4BB35lNRSUnyQ7Phwf2xictjbtdvj4D5D/Clz2wLBO
mRA4gQkBEzhQd4A7Mu7Q2EDtOFB3gNSwVEJ9nWgAeOQl5XHa57UxSmM6e/v50RtHSYoKMCvJXYjH
blv19PQQERFhOo5LYLNamBQXQ6Sf4A/vjdHGidHpyvS7Iy86ddqsmFkcqDswZjPO+h39HKo/xKxo
J7aepFTep0mLITReO+M05PEtRdS0dPPYquyxebPjIXis8wBMxzFMQvy8CfSx8bcPT3C4aowWzk27
HWrzla9hkjMuh+aeZsrbyjU0TDsKmwrp7u9m1jgnnEflbmXbatrtmtmlJbtKG/nX7pPcM38ScyaF
u9ocQ+PRzsNk+IT4eTEu2Jdvv3x4bE4dzLoFLF5KIHiYfBL3qM3VyipNOVB3AICcmJzhn3T4RfAK
gIwbNLJKOzp6+/nOq3lMigzgO8vGbpzKUzCdhwkAFiF4/JZsTjR08rt3j7vaHOfxD4e0ZZD/8rDn
m8cHxRPpF/nJRXiskVuXS2JwIpF+kcM7oa8Ljr0BmTeCz9gLMv9ycyGnW7v531un4edtble5GtN5
mHzCotQobp8bz993lpFb0exqc5xn+u3Q2QCl7w/rcCEEOTE55Nbljrm4x4BjgIN1B51L0S3apNTE
TF+jnWEa8eHxBl7cW8l9i5KYlRDmanNMMJ2H5jz55JMIISgsLPzkuYyMDMrLnd9nv/fee4mOjmbq
1M/ObNiyZQuTJ08mJSWFxx57bMT2/nBFBhPD/Pmv9Ydp6xljxYOp14B/JBx+ftinzIqZRX1XPdUd
1Roapj6lLaW029udcx6Hn4eQ+DHXQbexo5dvv3yE1OhAvrnUyRYsJpphOg+Nyc/PZ/r06WzatAlQ
ssDq6upITEx0Wtbdd9/Nli1bPvP8wMAADz74IG+//TYFBQWsW7eOgoKCEdkb6GPjD6unU9vWw//8
++jYuiO3eil31UWboe30sE4ZiheMta2rofqOYcc7mk5A2XaYsXZMtSORUvLdV/No67Hzp9tn4Otl
ble5Cx5b53E2P994jIJTbarKzJwQzE9vmHLJ4/Ly8vje977Hk08+yX//939TUFBAenr6iDLBFi9e
TEVFxWee37dvHykpKSQlJQGwevVq3nzzTTIzM53WATAjPoxvLU3jN1uLuTwtiltmxY1IjkuYdQ/s
+hMc/Bdc8b1LHp4UmkSoTygH6g5wU8rYmeF9oO4AEwImMD5w/PBOyP0nWGww6y5tDVOZZ3dVsK2o
np+vnEL6OCcHXZloyti5BRmjFBQUcOONN1JfX09rayv5+flkZ3+6D9GiRYuYPn36Z77ee++9Yemo
qalh4sT/9PaJi4ujpqZmVHZ/5fJk5iWF85M3j1LR2DkqWboSkQzJV8KBZ4Y1otYiLMyMnjmmMq6k
lByoOzD8LSt7Nxx6Xpl/EjROW+NUpPB0G798u4gr06O587IEV5tjcg6GWHkMZ4WgBVVVVURERODn
58fSpUvZunUreXl5ZGV9eu7Czp07XWLfxbBaBL/7/HSW/34nD607yKtfmT92tgxmfwnW3w7H3x5W
SmrOuBy2VW2jtrOWcQHuf3Etby2nuaeZnHHD3LI69m/oaVHelzFCZ28/X1t3iBA/L35zS7ZZs+WG
6L7yEEIsF0IUCyFKhRDfP8/r3xJCFAgh8oQQ7wshxuwtR35+/ieOYsWKFWzatEmTlUdsbCxVVVWf
/FxdXU1s7OhHcY4P8eO3t07jaE0bP994bNTydCN1GQTHwv6nh3X47HGzAdh7eq+WVqnG7tO7AZgd
M3t4J+x/GiLTIHGRhlaph5SS772WR1lDB3/4/HQiAn1cbZLJedB15SGEsAJPAEuBamC/EGKDlPLs
6O4hIEdK2SWE+Crwa2BMNuE5e5Vx+eWX8+Uvf5nu7m7VVx6zZ8+mpKSE8vJyYmNjWb9+PS++6Fyr
joDiJssAABuwSURBVAtxdWYMDy5J5okPTjAjPmxstL622pTYxwe/UALFERfvfzQ5bDJRflHsrNnJ
jSk36mTkyNlZs5PE4EQmBg/jb3HqMNTkwvLHx0wTxGd2VfBW3mm+u3wy81OGWcNiojt6rzzmAKVS
yjIpZR+wHvjU/1Yp5QdSyqH5qHuAMRSt/TRnrzx8fHzIzs7G29ub0FAnmtidxZo1a7jssssoLi4m
Li6Op59W7qxtNht//vOfWbZsGRkZGdx2221MmaLeVt23lk5mQUoEP37jKMdOtaomV1Nm3qkEiHP/
eclDhRAsjF3IrlO76HdcOk7iSrr7u9l/ej8LY4eZbpv7NHj5w7TV2hqmEgdONvPopkKuzojhK4vN
pofujN4xj1ig6qyfq4G5Fzn+i8Db53tBCHE/cD9AfLx7Nnh74YUXPvXzm2++OSp569ZduPXGihUr
WLFixajkXwirRfCH1TO4/o8f8dXnD7LhoQWE+rv51LagGEi/XgkUL/kf8Pa/6OGL4hbx79J/k9eQ
x8wYJ9qb68z+2v30OfpYFDuMLajuFsh/VZn17jeyGxY9qW/v4YEXDjIh1I/f3jbNnNHh5rhttpUQ
Yi2QA/zmfK9LKf8upcyRUuZERY3BORRjjMhAH564Yya1rcp/cPuAw9UmXZq5X1YCxYdfuOSh88bP
wyZs7Kxxv+SFs9lZvRM/m9/wmiHmPg32Lphzv/aGjZIe+wD3/+sArd12/rp2JiF+TozUNXEJejuP
GuDsjdq4wec+hRDiauB/gJVSyl6dbDO5BLMSwvjlqix2nWgaGwH0+Msgbjbs+uMl03aDvIOYHj2d
ndXu6zyklOys2cnccXPxsV4iiGzvgT1/U9KWxzs5olZnpJR8/7U8Dle18LvbpjNlQoirTTIZBno7
j/1AqhBikhDCG1gNbDj7ACHEDOBJFMdRr7N9JpfglllxfHlxEs/vqeRfuytcbc7FEQIWfANaKqHg
jUsevihuEcVniqnrrNPBOOcpbyunpqNmePGOI+ugs175/d2cv2w/wRuHT/HtpWlcmzXMokcTl6Or
85BS9gMPAVuBQuBlKeUxIcTDQoiVg4f9BggEXhFCHBZCbLiAOBMX8d3l6VyVHs3PNxaw43iDq825
OJNXKGmqH/9eGYR0EYYuyh+f+lgPy5zmo+qPAFgYdwnn4RhQVlsTZihDn9yYLUdr+c3WYlZOm8BD
V6a42hwTJ9A95iGl3CylTJNSJkspHx187idSyg2D318tpYyRUk4f/Fp5cYkmemO1CP6wZgap0YF8
9fkD5Fe7cQaWxQLzv64MiTqx7aKHpoamEuMf47ZbVztrdpIckkxs4CVqeAo3QnOZsupw4/Tc/RXN
/Nf6Q0yfGMqvzULAMYfbBsxN3JtAHxvP3juHUH9v7nlmHyeb3LiFSfZtEDReWX1cBCEEi+IWsfv0
buzDnAmiF132Lg7UHWBR3CWyrKRUfs/wJLce+HS8rp0vPrOf2FA//nn37LHTvcDkE0znYTJiYoJ9
efbeOQw4JHf+cx8N7W6a22DzgXlfhfIdUHPwoocujF1Ip72Tww2HdTJueOw9vRe7w37peEfFTjh1
COZ/DSzueUE+1dLNXf/ch4+XlWfvnUN4gJunfZucF9N5mIyKlOhAnr57NnVtPdz9//bR2u1ed+yf
MOse8A2B7RefdTJv/DxsFhs7qnfoZNjw2FGzA3+bPzOjL1KDIiV88EsIjHHbGeVNHb3c+c99dPT0
8+w9c5gYfvH6GxP3xXQeGqPmMChQZnfMmDGD66+//lPPqzUMaiTMjA/jr2tncbyunbv+uY92dxwi
5RsMC78JJVvh5K4LHhbgFcC88fPYWrEVh3SPWha7w877J99ncdxivKwXqX8oeRcqd8Pl3wUvX/0M
HCZnOvv+f3t3HldlmTZw/HcjCC4IKoIKJuJCbriA25hbWZpY1piOe4lTTlnzcWqamre30qzXtyxn
zJxyyaXeUtNqMk0t99xFnRJ3BURwAWSX7cC53z/OSVRADng2Dtf38+Hz4TzPwzmXtweucz/381wX
45ccIDE9l0WTIujQXEqsV2eSPGzMms2gAObNm0f79u1v2WbNZlBVNSjUnwXjuhOTlMnkZYe4XuCE
ZT56ToX6TWHLzDteeRUZEsnl65c5cvXOp7jsZd+lfaQXpBMZEln+QUYjbJ0JDVtBd+fr2ZGZZ2Di
0gPEpl5n8aQI+rRu7OiQxF2qESXZ2fiq6Woba2raGR6u+BO+NZtBJSYmsmHDBl577TXmzp17Y7u1
m0FV1UMdm/Lh2G68sPIoU1YcYulTPahb24neYrXrmhpErf8LnNkMoUPLPOz+FvdTx70OG+I2WF72
3IbWx67Hx9OHvs37ln9QzNdwNQZGfmrqqOhEsvINTFp6kDNXclg4KZx+baUihCuQmYeNWbMZ1PTp
03nvvfdwu62NqC2aQVXVsM7NmDu6Cwfj0pj06UHn64PebaLpSqStM033Q5ShrkddBrUYxI/xPzr8
qqvrhutsT9jOkJZDyj9lVVRoqiAc0Bk6/t6+AVYg7Xoh4xbv53hSJgvGd2dQqL+jQxJW4kQfC23I
ghmCLVizGdT69evx9/cnPDycHTt22Chi6xjRNRB3Nzemrz7KuMX7WTG5p/P0ZKjlAff/N6yNMhUN
7FJ2tf/IkEh+iPuBn5N+5v577rdzkCW2JWwjvzj/zqesjqyA9HgYv9ap+pNfycxn4qcHSEjLZfGk
CAbdK4nDlTjPO80FWbMZ1J49e1i3bh3BwcGMGTOGbdu2MWHCBMB2zaDuRmRYMxZNiuDs1RxGL9zH
lcx8h8Zziw6PQ9Mw2DYLCsu+P6VP8z408mrEhtgNdg7uVhtiNxBYP5Cu/l3LPiAvA3a+Cy37QpvB
9g3uDhKu5TJq4V4uZeSxIqqnJA5XpLWu9l/h4eH6didOnCi1zd5mz56tZ8yYobXWOj8/XwcHB+uA
gACdnp5+V8+7fft2HRkZeeOxwWDQrVq10rGxsbqgoECHhYXpmJiYSj2nrcZr//lU3fGNTbrP/2zR
Jy9n2uQ1qiR+j9ZvNtD6x9fLPeSd/e/o8M/DdVZBlh0DK5GSm6LDVoTpeYfnlX/Q99O1nuGrddJR
+wVWgf8kpOvwWT/pLjM366MJd/deF7YFROsq/t2VmYcNWbsZVHls3QzqbvQKaczqqb0p1ponPt7n
PLWwWv4Ouk2AvR/BlZgyD4kMiaSguIAtFyxrB2xtv10uXO4pq4sHTc2uev0JmpczM7GzH49f4Q+L
9uHl4caaqX3o2sL5+4iIqlG6gmJx1UFERISOjo6+ZdvJkydLXdIqymfr8bqcmcfkZYc4m5zDO491
YkxPJ2jglZsGH0WYFtCjfiy1XqC1JvLbSJrXb86Sh5bYPbxxG8ZhMBpY88ia0juLDbBwgKlfybQD
4Olt9/huprVm2Z54Zm04QVigD0ue7EETbydZ5xLlUkod1lpX6ZJCmXkIu2jmU4c1f+pD3zZ+vPrN
Md74LobCIgffhFe3ETz0DiQegiPLS+1WSjE8ZDgHLx8kISvBrqGdTjvNsdRjDA8ZXvYB+/8Fycfh
4fccnjjyDcW8vPZX3lp/goc6BLDqmT6SOGoASR7Cbry9PFj6ZATP9A/hs30XGLd4P8lZDl5I7zIG
gvvBTzMg63Kp3aNDR+Ph5sHy48vtGtbSmKXUda/LY20eK70zLc5UZiV0GLQvJ7nYSWJ6LqM+2cfa
w4n8+YG2fDw+nDq1nbOmlrAuSR7CrtxrufFfw9ozf2w3jl/KInL+bg7EXnNcQErB8H9CcSF883Sp
ez/86vjxaJtH+e7cd6TmpdolpKScJDbHb2ZUu1H4eN7WVa+o0HSZsZu7adbhQDvPpPDI/N3Ep15n
yaQIXnywnfQdr0EkeQiHeKRLc/49rS/1Pd0Zu3g/H/x42nF90f3aQOT7poq0u+aU2v1khycxGA18
efJLu4Tz2fHPUEoxocOE0ju3zoRLR2DER+DbovR+OygoKubt9Sd4culB/L29WPfCfQzuEOCQWITj
SPIQDhPa1Jv1L9zHyO5BzN92jtEL95FwLdcxwXQdD2FjTKeD4m6tqBvsE8zgloNZdXoV1w227VuS
np/ON2e/IbJVJE3rNb115+mNsO8j6PE0dBhh0zjKcy45h9//ay9LdscxqU9Lvnu+L6386jkkFuFY
kjyEQ9XzdGfOqC7MH9uNc8k5DJ23ixV74zEa7XwVoFIQ+QH4tYWv/wg5ybfsfqrjU2QXZrP2zFqb
hrHy1Eryi/OZ3GnyrTsyLsK3fzLd3PjQ2zaNoSxFxUYW7jxP5Ic/cykjjyWTInhrRCdp4lSDSfIQ
TuGRLs3ZNL0/EcGNeHPdcf6waB+xKTn2DcKzPoxaDvmZsHoiFJbMgsKahBEREMHnJz63Wb2rXEMu
K0+tZGDQQFr7ti7ZkZ8Fq8eb1mNGLbd7ufVTV7L4/cd7mb3xFAPaNWHz9P5ymkpI8hDOI9C3Dism
9+D9UV04fSWbofN+Zu6Pp8krLLuAoU0EdITHP4GLB2DtZNP9FGZRnaK4mnuVr89+bZOXXnlqJRkF
GUR1jirZaMiHVeNMNzI+8Sk0bl3+E1hZTkERszee5JH5u0lKz+Ojcd1YODEc/wbO1ytE2J8kDxuz
ZjOoqKgo/P396dSpU6l95TWDcmSTqKpQSvFEeBBbXhrA0I5N+XDbOQbP3ckPxy5jtxtaOz5uOoV1
ZhOse8HUKwNTi9peTXsx78g8rl6/atWXTMhK4ONfPmZQi0F08+9m2mgshm/+aFrIf+xjaDfEqq9Z
Hq01/z6axP3v72Dhzlge6xrITy8OYHhY8yq1EhAuqqp1TZzpy1lrW2mt9bRp03TXrl31nDlztNZa
5+Xl6YYNG2qj0Vjp59q5c6c+fPiw7tix4y3bi4qKdEhIiD5//vyN2lbHjx8vd3tZnGW8brf/fKoe
8o+duuUr6/WoT/bq6Phr9nvxHe+a6l9t/LvW5v+vhMwEHfF5hH5+6/NV+j8si9Fo1FGbonTvL3rr
KzlXTBuLi7X+7gXT6+9dYJXXscTec6l6xEe7dctX1utH5/+sj1xIs9trC/vjLmpb1YiS7O8efJdT
aaes+pz3NrqXV3q+UuFx1mwG1b9/f+Lj40ttL68Z1MCBA52iSdTd6BXSmPUv3MfKQxeZt+UsIz/e
x+D2Abw8JJTQpja+s7r/y5B7DfYvgNxUeHQ+LRq0YFrXaXxw+AM2X9jM0OCyG0pVxjdnv+HglYO8
0ecNAuoFmNZavn0GTn4P/f4KfZ6zwj/mzmKSMnlv82l2nUmhmY8X740M44nwILlvQ5SrRiQPR/qt
GdSsWbPu2AwqOzu71M++//77DB5ccZntsppBHThwoNzt1Y17LTcm9m7JyO6BLNsTzyc7zjPkn7sY
0jGA5wa2oYutiu8pBUP/F+r6mZotZSTAH75gQocJbIzfyOwDs+ndtDe+XlV//eTcZD6I/oCIgAhG
th0J2Vdg5Ri49B8YMht6P2vFf1Bp0fFpLNh+ju2nU/Ct68Frw9ozsU9LuYpKVKhGJA9LZgi2YM1m
UALq1nZn2qA2jOt5D8v2xrN8Txybj1+lX1s/ou5rxYC2Taz/SVkpGPAyNA6Bb5+FJQ/gPmo5b/3u
LcasH8Nb+99iTv851HKr/B9bg9HAm3vfpNBYyIzfzcAtMRrWPGXq0TF2JYQ+bN1/i1mxUbPtVDKL
f47lYFwajerV5q8PtWPS74Jp4OVcLWyF86oRycNRbm8G9cUXX3D58mUee+zWekV3O/MorxmUMzaJ
soaG9Wrz4oPteLpfK748kMCS3XFMXnaI4MZ1mdgnmCfCg/CpY+U/gp1Ggs89pktmFw8itNez/Dls
KnN/WcDre15nVt9ZlUogBqOBv+38G7uTdvN695doueufEL0MfIIgahM0C6v4SSop/Xohq6Mv8vm+
CyRl5NHcx4s3hndgTM8WztVrXlQL8o6xoZtnGQMGDGDq1Knk5eVZfebRo0cPzp49S1xcHIGBgaxa
tYovv/yS0NDQMre7Cm8vD6YOaM3kvq3YGHOZz/ZdYNb6E7y76RRDOjblifAg7mvjRy1rzUZa9IBp
B00lQvYvYHKDQAo7PcBHsd+j0bzd922LEoih2MDLu15ma8JWXmn+AKM3vWNaU+n9HAz6u1Wr5BYV
G9l5JoW1hxPZcvIqhmJN75BG/Hdkex7sEIB7LbngUlSNJA8bOnbsGCNHjgRKmkEdPXq0ys2gxo4d
y44dO0hNTSUoKIiZM2cyZcqUW5pBFRcXExUVdaMZVHnbXUltdzdGdA1kRNdAYpIy+Sr6It/95xLf
/3KJgAaePNypGcM6NyO8ZcO7TyR1fGH4P6DLWPh+OlP3foZbk2Z8GLue4oJsXu8/G+/a5f/xz8jP
4I0dL7H96kFezSpgfNwyaNYVxq+xWkOnomIjB+PS+CHmMptirpCaU0jjerWZ2DuY0T2CuLdpA6u8
jqjZpBmUAFxvvAqKitl2Mplvjyax40wKhUVGmnh7cn+oPwNDm9C3rd/dn983GuH8Vjj0KUuS9zKv
oQ/eRs1YjwAmBA6koXfJKcLUzIt8dmkHq4tSyFPwalom4wIHQMQUaDWgVCOqysrMNbDrbAo7z6Sw
/VQy164XUsejFoPubcLj3YIYGNoED5lliNvcTTMoSR4CcO3xyikoYvupZDbFXGHXmRSyC4qo5abo
1sKXXiGN6NWqMeEtG1LP8y4m4hkJnDjyKUsubeen4nTqGI0EG4oA0ECchzsG5cZQDz/+GPQAbbtF
QYPmVX657HwD0RfSORCbxoG4a/xyMQOjBp86Hgxo14RhnZsyoJ2/9NYQdyTJQ5LHXasp42UoNnLk
Qjo7zqSw9/w1YpIyKTZqarkp2vrXJyzIh85BvnRo1oC2AfWrNDs5n3Ge/zu2jNTckrvQA+o3Z2Ln
KFo2aFnp58vMM3D2ajYnLmfxy8VMjiVlcC45B6MGdzdFWJAPfdv4MTC0CV2CfGUdQ1jsbpKHS695
aK2lnIIFXOEDhKU8arnRK6QxvUIaA3C9oIgjCekcikvj16RMtpxM5qvoxBvHN/Pxoo1/fe5pVJd7
GtWlRaO6NPXxwt/bkybenni6l/5k39q3NW/2s7zybb6hmJTsApKzC7iSmU9CWq756zrnknO4mlVw
41i/+rUJC/JlWOdm9AxuRLd7GsrsQjiEyyYPLy8vrl27RuPGjSWB3IHWmmvXruHlVTOL3dXzdKdf
2yb0a9sEMI1HYnoep69kcyY5m7NXczifksOGY5fJyC1dTdfbyx2fOh741PHA28sdL49aeLq74ele
65bF+WKjpqComIIiI3mFxWTnF5GZZyArz0B2QVGp521UrzYtGtWlbxs/2gV40y6gPvc2bUAzHy95
Pwun4LKnrQwGA4mJieTnO7hHdjXg5eVFUFAQHh5yg9idZOYZuJiWS3J2PslZpplC2vVCsvIMZOUb
yMorupEg8g3F3NySxE1Rklg8atHAy50G5qTTuF5t/L29aNLAkwBvL1o0qoO33Kwn7EBOW5XBw8OD
Vq1aOToM4UJ86njgE+gD+FR4rBCuzu4ra0qpoUqp00qpc0qpV8vY76mUWm3ef0ApFWzvGIUQQtyZ
XZOHUqoWsAB4GOgAjFVK3V7idQqQrrVuA/wDeNeeMQohhKiYvWcePYFzWutYrXUhsAoYcdsxI4AV
5u/XAg8oWSEUQginYu81j0Dg4k2PE4Fe5R2jtS5SSmUCjYHUmw9SSj0DPGN+WKCUirFJxNWPH7eN
VQ0mY1FCxqKEjEWJ0Kr+YLVdMNdaLwIWASiloqt6xYCrkbEoIWNRQsaihIxFCaVUdMVHlc3ep62S
gBY3PQ4ybyvzGKWUO6ZLW67ZJTohhBAWsXfyOAS0VUq1UkrVBsYA6247Zh3wpPn7J4Bt2hVuRhFC
CBdi19NW5jWM54HNQC1gqdb6uFLqLUyN2NcBnwKfK6XOAWmYEkxFFtks6OpHxqKEjEUJGYsSMhYl
qjwWLnGHuRBCCPuS8ptCCCEqTZKHEEKISqtWyUNKm5SwYCxeVEqdUEr9qpTaqpSqfCOJaqKisbjp
uJFKKa2UctnLNC0ZC6XUaPN747hSynWa2t/Ggt+Re5RS25VSR82/J8McEaetKaWWKqWSy7sXTpl8
aB6nX5VS3S16Yq11tfjCtMB+HggBagO/AB1uO+Y54BPz92OA1Y6O24FjMQioa/7+2Zo8FubjvIFd
wH4gwtFxO/B90RY4CjQ0P/Z3dNwOHItFwLPm7zsA8Y6O20Zj0R/oDsSUs38YsBFQQG/ggCXPW51m
HlLapESFY6G13q61zjU/3I/pnhpXZMn7AmAWpjpprlyj35KxeBpYoLVOB9BaJ9s5RnuxZCw00MD8
vQ9wyY7x2Y3WehemK1fLMwL4TJvsB3yVUs0qet7qlDzKKm0SWN4xWusi4LfSJq7GkrG42RRMnyxc
UYVjYZ6Gt9Bab7BnYA5gyfuiHdBOKbVHKbVfKTXUbtHZlyVjMQOYoJRKBH4AXrBPaE6nsn9PgGpc
nkRYRik1AYgABjg6FkdQSrkBc4GnHByKs3DHdOpqIKbZ6C6lVGetdYZDo3KMscByrfUHSqk+mO4v
66S1Njo6sOqgOs08pLRJCUvGAqXUYOA14FGtdcHt+11ERWPhDXQCdiil4jGd013noovmlrwvEoF1
WmuD1joOOIMpmbgaS8ZiCvAVgNZ6H+CFqWhiTWPR35PbVafkIaVNSlQ4FkqpbsBCTInDVc9rQwVj
obXO1Fr7aa2DtdbBmNZ/HtVaV7kgnBOz5Hfk35hmHSil/DCdxoq1Z5B2YslYJAAPACil2mNKHil2
jdI5rAMmma+66g1kaq0vV/RD1ea0lbZdaZNqx8KxmAPUB9aYrxlI0Fo/6rCgbcTCsagRLByLzcBD
SqkTQDHwstba5WbnFo7FS8BipdRfMC2eP+WKHzaVUisxfWDwM6/vvAl4AGitP8G03jMMOAfkApMt
el4XHCshhBA2Vp1OWwkhhHASkjyEEEJUmiQPIYQQlSbJQwghRKVJ8hBCCFFpkjyEEEJUmiQPIYQQ
lSbJQwghRKVJ8hBCCFFpkjyEEEJUmiQPIYQQlSbJQwghRKVJ8hBCCFFp1aYkuxDOTikVAgwGfIHW
WuupSqm/ARnAg1rrUQ4NUAgrkpLsQliBUsoXGK21XmR+/JN51yhMCWUN0LCGtnsVLkhOWwlhHTcS
h1kj4IjWOkNrvRbTTEQSh3AZMvMQwgqUUr43JwellMZ0qmqLA8MSwmYkeQhhZUqpwcBPWmvl6FiE
sBU5bSWE9T0IHHF0EELYkiQPIazAfKXVbwYD0Tft8zXPRoRwGZI8hLhL5sRwXikVopTqbt588+L4
M7L2IVyNrHkIcZfMs45XgMPmTV8B7978WK60Eq5GkocQQohKk9NWQgghKk2ShxBCiEqT5CGEEKLS
JHkIIYSoNEkeQgghKk2ShxBCiEqT5CGEEKLS/h8EpRFpog+YQAAAAABJRU5ErkJggg==
){:width="100%"}


The first thing we notice is that these curves are symmetric across $x=0.5$.
This makes sense since we're not testing for whether our coin is biased
specifically towards heads or tails, but just whether it's biased in some way.
So a bias of landing on heads 90% of the time and a bias of landing on tails 90%
of the time are treated equally.

The next interesting detail is that the curves start out very fat, for $N = 10$,
and get sharper and sharper as $N$ increases. That just means that as we do more
coin flips in our experiment, we're able to detect more subtle biases, i.e. when
$x$ is closer to being 0.5. Let's make this same plot again for larger values of
$N$.


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
    plt.plot(X, curve_dict[k], label=r'$N = {0}$'.format(k))
plt.legend(loc='lower left')
plt.xlabel(r'$x$', size=20, usetex=True)
plt.ylabel('power', size=16)
plt.axis([0, 1, 0, 1])
{% endhighlight %}




    [0, 1, 0, 1]




![png](data:image/png;base64,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
bwAAAABJRU5ErkJggg==
){:width="100%"}


The curves continue to get narrower as we increase $N$. Although I'm still a
little surprised that the curve for $N=1000$ doesn't look like an upside-down
delta function; this plot implies that if you suspect a bias where $x=0.55$, you
still can't be absolutely sure you've done enough testing to catch it _after
1000 coin flips_!

There's something else that's surprising in the first plot above. Look at the
lowest points on each of the curves, right in the middle where $x=0.5$. This
point doesn't always get higher as you increase $N$! When I first took a glance
at these plots, I came away with the pattern that "as you increase $N$, the
curves get skinnier and the nadir gets higher." But in the first plot, the nadir
is highest for the curve where $N=40$, which is not the skinniest or the fattest
curve! That means the relationship between the height of the nadir and the width
of the power curve, or $N$, may not be as simple as we might think.

## Plots with fixed x

Let's plot the value of the power when $x=0.5$ for different values of $N$. You
can interpret this value as the probability that you'd conclude that you had a
rigged coin, when in fact the coin wasn't biased at all.


{% highlight python %}
N = [10, 30, 60, 90, 150, 250, 400, 700, 1000]
z = [power(0.5, n) for n in N]
{% endhighlight %}


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
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
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
gg==
){:width="100%"}


Clearly there's some non-monotonic behavior here. This is interesting. I would
have thought that the power at $x=0.5$ would be a strictly decreasing function
of $N$--in other words, that there'd be less of a chance of drawing the
incorrect conclusion that the coin is biased, as we do perform more trials. This
is just based on the intuition that we usually have a higher probability of
drawing correct conclusions as the number of trials goes up.

Let's zoom in on the region from $N=0$ to $N=200$.


{% highlight python %}
N = np.arange(10, 210, 5)
z = [power(0.5, n) for n in N]
{% endhighlight %}


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
{% endhighlight %}




    <matplotlib.legend.Legend at 0x1079e6e10>




![png](data:image/png;base64,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
H1pKqc/lW10fAAAAAElFTkSuQmCC
){:width="100%"}


That's definitely interesting behavior. Let's actually plot the power for every
value of $N$:


{% highlight python %}
N = np.arange(10, 210, 1)
z = [power(0.5, n) for n in N]
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
{% endhighlight %}




    <matplotlib.legend.Legend at 0x107615ad0>




![png](data:image/png;base64,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
Vhl3AAAAAElFTkSuQmCC
){:width="100%"}


The overall trend of the curve seems to be going up, but there's some
interesting periodic behavior here. On the right you can see a series of
sawtooths, each of which is slightly higher than the previous one and whose
periodicity seems to be increasing (but not by a very clear pattern; see for
yourself below).


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
plt.axis([240, 311, 0.035, 0.053])
{% endhighlight %}




    [240, 311, 0.035, 0.053]




![png](data:image/png;base64,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
){:width="100%"}



{% highlight python %}
%matplotlib notebook
N = np.arange(10, 510, 1)
z = [power(0.5, n) for n in N]
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
plt.show()
{% endhighlight %}


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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HBAHigITJ+wqBXPRjb+vTrhkSTh4t81TJaQgT7t6WbXo9pXVJTffNZadgcLzTDuWaFyfrJBVdPcD8aSKNj/R4qokBJumPJvi8/zlvnCgcGAyOVCmEF296j2w79j5dw/Zobu4NqerpCA/b2f56fMmbcIXKVoD4zquqal+VpHWdFSeUa9ll96oHCz5J4UDJ598cvXBD36wevTRR6t3v/vd1SGHHFJLGut+bF5ZbbXVqqc+9anVhRdyENLkuWKBuTqVgpHy0iOdKqH7mV5lGeWnjbcFX+XSFOK4pijB02T9NMWrbDO5rsp3smZSvpKncGBVcACl9d73vrc66aSTqquuuqo69thjw7UJ95lnnlldcsklE6u8oLsoMFf72mThojrBNgSzNjx0gLpAG/AF7uTLbw3Bb5xzw1imKAHetBGkKV5lm8kVi25cU6EzKU/JM7kcYGqcZYMFh/y01b7zspe9rDr11FMD4z75yU9W73//+4dm4gUXXFBtv/321XbbbVfNmzevOuigg6rjjz9+aDiTlKFMIbralAXmojrBNgSzLLAOUBdoAz7g6ICHnsD5ydGNY4oSyKv63aVxTYUmNpVL4UDoO/59vDb7zuGHH14deuih1W233VZdfPHF1Qkn9H6kY6+99qruueeeKbVwxBFHVC9/+ctD/C233FJttdVWnTS8hHz++ed37n2AHYaveMUrwrtvBx98cPWe94TPJPokExEuCsxVoxTMPDvmyR8Y28bpGKCRguR4qgcefqyDuS34AGRtysMmTlOUba5NCVbbL2FT3uIKB8bBgcN/fGV11ZK7G0FffNNdPf2ehPSdj/zgsurYC26qzbfz09ev/sZO65nO7b333rYp7PHqyCOPrM4666ywNuXznH322f525PAvf/nL6hnPeEZQmPvuu2+14447VpRh0lxRYK5GdbL6e/ZeUP3o4iWtnyohBfm3B+xSfem0ha3Dh5SmqcimeEf+0MFiFQ3NspLhScwBP2j1xWyK92mmC19++eXVrbfeGl4eZnNF7gaxwFBIN998cycrp4wQV+cUv9lmm1Wvf/3rK6YfiwKr49QExWkN7KXP3qz68Ct3bJ0yba54/fPmVwc+vzsV0CYipiKZ+shdW1OUOdxyXzgwWzgwnaXE2ldd3+FDsN89+EUzJhPF9ba3vS2sV33gAx+o2Em433779cAbxALbY489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" width="432">


As you can see, the overall trend of slowly rising sawtooths with irregular
periods keeps on going. I wouldn't have predicted this. I'm not just referring
to the interesting shape of the graph, but the fact the the power is essentially
increasing with $N$ when $x=0.5$. This sounds good, until you realize that when
$x=0.5$ the power is the probability of drawing the _wrong_ conclusion.

## Minimum sample size needed

We can also answer the question: given a p-value, and a suspected minimum bias
of the coin, what is the minimum value for the number of flips $N$ we need to
carry out so that the probability of concluding the coin is rigged is at least
0.8?

We can do this by fixing $x$ at the suspected bias value, and calculating the
power over a large range of $N$. Then we can just look for the first value of
$N$ for which the power is at least 0.8.


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

## Testing

Another nice thing about examining a simple situation like coin flipping is that
we can do simulations to check our function. Since the output of our function is
a probability, we can use brute force to calculate it by running many trials,
using a random number generator, and seeing the proportion for which our
condition is satisfied.

In this case, since the power tells us the probability of making a conclusion in
an experiment, for every trial we have to simulate an experiment, where we flip
a coin $N$ times and use a $p$-value to conclude whether or not it's biased.
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

---


{% highlight python %}
N = np.arange(10, 210, 5)
z = [power(0.4, n) for n in N]
{% endhighlight %}


{% highlight python %}
plt.plot(N, z, 'o-', label=r'$x = 0.5$')
plt.xlabel(r'$N$', usetex=True, size=20)
plt.ylabel('power', size=16)
plt.legend(loc='lower right')
{% endhighlight %}


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAAAXNSR0IArs4c6QAANF1JREFUeAHtnQl4FdXZx98QAgHZIcgqRJBN2QyiFrUu7CootRa1n7UbtgVtq0WxtYjYVtRqtVW/FvtZrU+R4o4biIJWrSJBZAdBQEhYEmSVLGT7/u/lDkxuZm62ufeemft/n+fPzJzZ3vM7IW/OmbOI0EiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEiABEjAKAIpRnnjoTNt27at6N69u4dP5KNIgARIIPgEli9fvhe5zPBDThv6wcm6+KjBKzs7uy638h4SIAESSFoCKSkpX/ol8w384ij9JAESIAESIAE7AQYwOw3ukwAJkAAJ+IaAKQFsNIhthDZD0xzonYK0JdAKaBU0FqKRAAmQAAkkMQETAlgq+D8GjYH6QdeEt9gctzuxNw8aDE2EHodoJEACJEACSUzAhE4cQ8Ffa15bwuUwF9vx0LrwsW4qoBbh45bY7gzvc0MCJEACviHw8opceWDhRtl5oFA6tWoiU0f1lisGd/aN/6Y5akIA09LbYQOTg/2zbce6OwN6C7oJOgkaDtFIgARIwDcENHjd8eJqKSwpC/mciyCmx2pWEGOAC+Go8T8mNCHWxFltVnwK6gLp969nICffJyFd+85n5+fnY0MjARIgATMIaM3LCl6WR3r8wMINoUMrwGlg0yYnK8BpumW6P2zWYsmc9npoaz9nXZNMWxNqYFo6XW3QNUidKLFjJ36I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" width="432">





    <matplotlib.legend.Legend at 0x10800d310>



Does this imply that you have to think about adjusting $p$-values as you
increase $N$? Is that correct, wrong, deep, or obvious? I'm not sure which of
those apply.

---

[1] Coin flipping is like mouse models in biology. I heard of a cancer
researcher once saying something like "If you can't cure cancer in a mouse,
you'd better find another job" (since you have no chance of curing it in
people). Similarly, if we can't apply statistical tools to a binomial model like
coin flipping, then we probably don't know what we're talking about.
<hr><br />
[Source Notebook File](https://github.com/gte620v/gte620v.github.io/tree/master/_ipynb/power_analysis.ipynb)