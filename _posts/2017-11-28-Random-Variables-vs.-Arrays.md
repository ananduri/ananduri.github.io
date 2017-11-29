# Random variables vs. arrays
Let's go over the correspondence and distinction between _random variables_ and vectors. This is a detail that is glossed over in many treatments of ICA, and in explanations of statistical methods in general. It's not that it's tricky; I just haven't seen this relationship made explicit before, so I want to do that here.

A _random variable_ is the fundamental abstraction of statistics. For the purposes of our discussion about ICA, think of them as numbers _drawn from some probability distribution_. By saying this, we've stipulated that: 

1) Random variables don't take on a definite value. The closest we can get is a particular _measurement_ of a random variable, which is a regular, non-random variable with a fixed value.

2) There exists some probability distribution from which you (or Nature) draws numbers when you're making a measurement of the random variable. That number is then bound to the non-random variable that represents the particular measurement.

To distinguish between random and non-random variables, we're going to write an $R$ next to random variables, like this: $X^R$.

So insted of taking on a certain value, a random variable $X^R$ stands for the process by which you'd generate a measurement, $x$, of this random variable. And, when we talk about $X^R$, we're also implicitly referring to the probability distribution behind it.

Since we want to translate math into code, how are we supposed to represent random variables in a program and use them to actually make calculations? We can't initialize $X^R$ with some fixed value, like we would for regular variables. But, since $X^R$ represents a process, maybe we could use something like a Python generator to represent it, where every time it yielded a value, it calculated a new draw from the probability distribution behind $X^R$ (which we'd have to supply). But here's a bigger problem: what if we don't know the probability distribution?

This is precisely the case in our ICA discussion. The criterion we've specified for our transformed variables, the sources, is that the sources generate independent signals, and we've mathematically described this by saying that the transformed data is drawn from a probability distribution that factorizes. Implicitly, we're thinking of the signals we want to identify as random variables. But if the signals from each source are random variables, the measurements made by our microphones, which are just a linear combination of the signals, must also be random variables. And we aren't given any probability distributions, we just have a lot of observed data points! How can we bridge the gap between the abstract objects we need to work with (probability distributions), and the actual information sitting in our computer (observed data)?

Let's talk about the available information in our computer. Consider the measurements we took from one microphone. While we were recording data, we measured one number at a time for a bunch of time points in that interval. Putting these numbers together in the order that we measured them, we stored these measurements as some array. This is our data.

Could we use this array of measurements to figure out the probability distribution, and then go back to thinking in terms of random variables? We might be able to, with techniques like kernel density estimation. But that seems too complicated; can we work more directly with the data? This brings us to the key question we need to answer: 

**How can we work with an array of data instead of a random variable?**

Well, each measurement, each element of the array, represents a draw from the hidden probability distribution. If we have a lot of measurements, then the data should be almost as good as having the distribution itself. Furthermore, notice that in all the calculations we'll need to do, the bare random variable doesn't appear--there's always some attribute of the distribution, or some function of the random variable, that is involved. For example, in the formulas we'll see below, the random variable representing the signal from source 1, $s_1^R$, never appears by itself. Rather, we only see things like the mean of $s_1^R$, or the covariance between two sources $\mathrm{Cov}\, (s_1^R, s_2^R)$. Why don't we try to use our data to directly calculate these functions of the random variable?

<!-- (have this question appear in a header--want to show nested considerations/questions we're answering, so people don't lose track of the current objective) -->


<!-- have a chart explaining what each variable means -->


For instance, we could estimate the mean of $x_1^R$, the random variable representing the signal from microphone 1, by finding the mean of all the measurements we took from microphone 1 [1]. Since those measurements are represented by the vector $\mathbf{x}_1$, we can write this as

\begin{equation*}
\mathrm{mean}\, (x_1^R) = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_1\mathrm{[i]}.
\end{equation*}

That might look weird, since I'm mixing math and coding notation. But I think this works, because most people recognize $\mathrm{[i]}$ as an array access, which is exactly what we mean here, and the mathematical way of writing this would be to use another subscript, which is confusing. Also, putting summation symbols everywhere is kind of a pain, though, so let's use some math notation to make this concise: angle brackets $ \langle \; \rangle$.

\begin{equation*}
\frac{1}{n} \sum_{i=1}^n \mathbf{x}_1\mathrm{[i]} = \langle \mathbf{x}_1 \rangle
\end{equation*}

The $ \langle \; \rangle$ represent _an average of the thing inside_. You can also go backwards; if you see $ \langle \; \rangle$ around something, you can slip in a $\mathrm[i]$ next to each array, and write $\frac{1}{n}\sum_{i=1}^{n}$ on the left, taking the average. 

Now let's try to estimate the covariance between two of the signals, $x_1^R$ and $x_2^R$. If we had access to the probability distributions, we could start to calculate the covariance by drawing one sample from each of the distributions at the same time, and multiplying them together. After doing many of these simultaneous draws, we could then average all the products we got. This number would be the covariance. Of course, we don't have access to the distributions, so we can't do this.

Fortunately, we can use the data we picked up from the microphones to simulate this process. Since the microphones all recorded data for the same time points, we just need to multiply the datum microphone 1 recorded by the datum microphone 2 recorded _at the same time_, and average that product over all the time points. Using the new notation, the covariance between two random variables looks like:

\begin{equation*}
\mathrm{Cov}\, (x_1^R, x_2^R) = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_1\mathrm{[i]}\,\mathbf{x}_2\mathrm{[i]} = \langle \mathbf{x}_1\mathbf{x}_2 \rangle
\end{equation*}

This illustrates an important rule: when you have more than one array inside the $ \langle \; \rangle$, put the array access $\mathrm{[i]}$ next to all of them, _with the same index_. This is important. Interpreted in terms of random variables, it means you're considering the values of both variables _at the same time point_. To compare their values at different times, or equivalently to compare values of the data arrays at different indices, wouldn't make sense [2].

Hopefully, now you see how we can use the data available to us, on the computer, to pretend that we're working with the abstraction of random variables that statistics relies upon. The random variables themselves never appear in any calculation, only functions of them like the mean or covariance. That means we can use our surrogates for each random variable, an array of data, to estimate those functions directly, and insert the values we get into our program. 



---

[1] They're not the same thing, because in the language of orthodox statistics, the mean of $x_1^R$ is a population mean, while the mean of the measurements we took is the sample mean. Orthodox statistics spends a lot of time focusing on when and how we can identify the two.

In Bayesian language, there's no such thing as a random variable $x_1^R$--there is only our state of knowledge about the mean of $x_1$, which we're progressively improving by collecting more measurements. And given certain prior information, our best guess of the mean of $x_1$ is the mean of the measurements we took.

I'm identifying how both approaches would talk about what we're doing because I want to remain aware of both--I intellectually prefer the Bayesian way of doing things but out of habit it's usually easier to describe things from an orthodox viewpoint.

[2] One more comment about notation: sometimes the angle brackets $ \langle \;  \rangle$ are written directly around the random variables. This is a just a cleaner way of writing the mean: $\mathrm{mean} (x_1^R) = \langle x_1^R \rangle$, or the covariance: $\mathrm{Cov}(x_1^R, x_2^R) = \langle x_1^R x_2^R \rangle$. When you see $ \langle \; \rangle$ around random variables, the whole expression is sometimes called an _expectation value_, because it's what you would expect to measure for whatever's inside, given your data set.
