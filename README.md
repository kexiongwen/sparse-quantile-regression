# sparse-quantile-regression

##  Model setting	

Assume that $y_{i}=x_{i}^{T}\beta+\epsilon_{i}$ with $\epsilon_{i}$ being i.i.d random variables from the skewed Laplace distribution with density



$$
f(\epsilon)=q(1-q) \exp[-\rho_{q}(\epsilon)]
$$



for $q \in (0,1)$ . Then the joint distribution of $Y=(y_{1},...,y_{n})$ given $X=(x_{1},...,x_{n})$ is



$$
f(Y\mid X, \beta)=q^{n}(1-q)^{n}\exp\left[-\sum_{i=1}^{n}\rho_{q}(y_{i}-x_{i}^{T}\beta) \right]
$$



Since the skewed Laplace distribution can be represented as a scale mixture of normals, we have



$$
y_{i}=x_{i}^{T}\beta+(\theta_{1}w_{i}+\theta_{2}z_{i}\sqrt{w_{i}})
$$



where $\theta_{1}=\frac{1-2q}{q(1-q)}$,  $\theta_{2}=\sqrt{\frac{2}{q(1-q)}}$,  $z_{i}\sim N(0,1)$ and $w_{i} \sim \mathrm{Exp}(1)$.



We consider a $L_{\frac{1}{2}}$ prior on $\beta_{j}$ such that

 

$$
\pi(\beta_{j} \mid  \lambda)\propto \exp[-\lambda |\beta_{j}|^{\frac{1}{2^{\gamma}}}]
$$



with hyper-prior 



$$
\lambda \sim \operatorname{Gamma}\left(\frac{1}{2}, \frac{1}{b}\right)
$$



We consider a full Bayesian formulation of the exponential power prior, which introduces a non-separable bridge (NSB) penalty



$$
\operatorname{pen}(\beta)=-\log \int \prod_{j=1}^{P} \pi\left(\beta_j \mid \lambda\right) \pi(\lambda) d \lambda=(2^{\gamma}p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)
$$



with the hyper-parameter $\lambda$ being integrated out.



## The EM algorithm for posterior mode search 

Considering the quantile regression with non-separable bridge penalty, we have the following objective function


$$
\arg \min_{\beta}\sum_{i=1}^{n}\rho_{q}(y_{i}-x_{i}^{T}\beta)+(2^{\gamma}p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)
$$



This is equivalent to finding the mode of the posterior of the $\beta$.
$$
\arg \max_{\beta}\log \pi(\beta \mid Y)
$$
where $\pi(\beta \mid Y) \propto f(Y \mid \beta, X)\pi(\beta)$. By using the scale mixture of normal representation of the skewed Laplace likelihood,  we have


$$
f(Y,W \mid \beta,X) \propto \prod_{i=1}^{n}w_{i}^{-\frac{1}{2}}\exp\left\{-\frac{1}{2}\sum_{i=1}^{n}\left[\frac{(y_{i}-x_{i}^{T}\beta)^{2}}{\theta_{2}^{2}w_{i}}+\frac{\theta_{1}^{2}w_{i}^{2}}{\theta_{2}^{2}}-\frac{2\theta_{1}(y_{i}-x_{i}^{T}\beta)}{\theta_{2}^{2}}\right]\right\}
$$



where $\theta_{1}=\frac{1-2q}{q(1-q)}$ and $\theta_{2}=\sqrt{\frac{2}{q(1-q)}}$ . The above data augmentation strategy allows us to tackle the optimization problem with EM algorithm. By treating $W$ as the missing data, we have the following complete-data surrogate objective function:



$$
\begin{aligned}
Q(\beta \mid \beta^{(m)}) & =\mathrm{E}_{\pi(W \mid Y,\beta^{(m)})} \left[\log \pi(\beta, W \mid Y)\right]\\
                          & =C -\frac{1}{2\theta_{2}^{2}}\sum_{i=1}^{n}E_{\pi(w_{i} \mid y_{i},\beta^{(m)})}[w_{i}^{-1}](y_{i}-x_{i}^{T}\beta)^{2}+\frac{\theta_{1}}{\theta_{2}^{2}}\sum_{i=1}^{n}(y_{i}-x_{i}^{T}\beta)-(2p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)\\
                          & =C^{\prime}-\frac{1}{2\theta_{2}^{2}}\sum_{i=1}^{n}\delta_{i}^{(m)}\left[y_{i}-x_{i}^{T}\beta-\frac{\theta_{1}}{\delta_{i}^{(m)}}\right]^{2}-(2p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)\\
                          & =C^{\prime}-(T^{(m)}-X\beta)^{T}\Lambda^{(m)}(T^{(m)}-X\beta)-(2p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2^{\gamma}}}+1/b\right)
\end{aligned}
$$



where $\delta_{i}^{(m)}=E_{\pi(w_{i} \mid y_{i},\beta^{(m)})}[w_{i}^{-1}]=\frac{1}{q(1-q)|y_{i}-x_{i}^{T}\beta^{(m)}|}$, $T^{(m)}=Y-(1-2q)|Y-X\beta^{(m)}|$ and $\Lambda^{(m)}=\mathrm{Diag}\left(\frac{1}{4|y_{i}-x_{i}^{T}\beta^{(m)}|}\right)$.  The constant $C$ and $C^{\prime}$ absorbs all the summands that do not depend on the parameters of interest $\beta$. 



Therefore, at iteration $m+1$, we need to solve 



$$
\beta^{(m+1)}=\arg \min_{\beta}(T^{(m)}-X\beta)^{T}\Lambda^{(m)}(T^{(m)}-X\beta)+(2p+0.5)\log\left(\sum_{j=1}^{p}|\beta_{j}|^{\frac{1}{2}}+1/b\right)
$$



by using our coordinate descent algorithm.



One potential probem is that the EM algorithm is very sensitive to the intial value becase $\Lambda_{ii}^{(m)}=\frac{1}{4|y_{i}-x_{i}^{T}\beta^{(m)}|}$. A small $|y_{i}-x_{i}^{T}\beta^{(m)}| $ will lead to a large weight given to $(x_{i},y_{i})$. Especially when the dimensional of the model is large, lots of the weight could be large due to the overfitting. These will reduce the effect of sparse penalty. 

 A heuristic approaches to solve this issue are to set a upper bound to the weight such that



$$
\Lambda^{(m)}=\mathrm{Diag}\left(\frac{1}{4\max(\epsilon,|y_{i}-x_{i}^{T}\beta^{(m)}|})\right)
$$



and use $\beta^{(0)}=0$ as initialization.