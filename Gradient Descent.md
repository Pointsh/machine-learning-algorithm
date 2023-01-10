# Gradient Descent(경사 하강법)

* 경사하강법을 배우기 전 미분(differentiation)을 짚고 넘어가야합니다.

![image](https://user-images.githubusercontent.com/44185037/211506599-c9e122c9-8279-4439-8174-cb53440b7f53.png)

미분이란? **'함수 f의 주어진 점 (x,f(x))에서의 접선의 기울기를 구한다.'**입니다.

즉, 변수의 움직임에 따른 함수값의 변화를 측정하는 함수 라고 보시면됩니다.

최적화에서 가장 많이 사용하며 sympy library를 사용하면 표현이 가능합니다.

```python
import sys as sym
from sympy.abc import x

sym.diff(sys.poly(x**2 + 2*x+3),x)
```
# Gradient Descent 방법의 직관적인 의미

* Gradient Descent 방법은 steepest descent 방법이라고도 불리는데, 함수 값이 낮아지는 방향으로 독립 변수 값을 변형시켜가며 최종적으로는 최소 함수 값을 갖도록하는 독립 변수 값을 찾는 방법입니다.

* steepest descent 방법은 다음과 같이 많이 비유되기도 합니다.

**앞이 보이지 않는 안개가 낀 산을 내려올 때는 모든 방향으로 산을 더듬어가며 산의 높이가 가장 낮아지는 방향으로 한 발씩 내딛어갈 수 있다.**

* Gradient Descent는 함수의 최소값 즉 최소값의 위치를 찾는 문제에서 활용됩니다. 

* 함수의 최소, 최댓값을 찾으려면 “미분계수가 0인 지점을 찾으면 되지 않느냐?”라고 물을 수 있는데,

* 미분계수가 0인 지점을 찾는 방식이 아닌 gradient descent를 이용해 함수의 최소값을 찾는 주된 이유는 우리가 주로 실제 분석에서 맞딱드리게 되는 함수들은 닫힌 형태(closed form)가 아니거나 함수의 형태가 복잡해 (가령, 비선형함수) 미분계수와 그 근을 계산하기 어려운 경우가 많고, 실제 미분계수를 계산하는 과정을 컴퓨터로 구현하는 것에 비해 Gradient Descent는 컴퓨터로 비교적 쉽게 구현할 수 있기 때문입니다.

* 데이터 양이 매우 큰 경우 gradient descent와 같은 iterative한 방법을 통해 해를 구하면 계산량 측면에서 더 효율적으로 해를 구할 수 있습니다..

* gradient descent는 함수의 기울기(즉, gradient)를 이용해 x의 값을 어디로 옮겼을 때 함수가 최소값을 찾는지 알아보는 방법이라고 할 수 있습니다..

![image](https://user-images.githubusercontent.com/44185037/211508080-256a0432-8e8c-4aea-94fa-93ad7c362b16.png)

출처 : https://datahacker.rs/gradient-descent/

* 기울기가 양수라는 것은 x값이 커질 수록 함수 값이 커진다는 것을 의미하고, 반대로 기울기가 음수라면 x값이 커질 수록 함수의 값이 작아진다는 것을 의미한다고 볼 수 있습니다.

* 또, 기울기의 값이 크다는 것은 가파르다는 것을 의미하기도 하지만, 또 한편으로는 x의 위치가 최소값/최댓값에 해당되는 x좌표로부터 멀리 떨어져있는 것을 의미하기도 합니다.

* 이 점을 이용해서 기울기가 양수라면 음의 방향으로 x를 옮기면 되고, 기울기가 음수라면 양의 방향으로 x를 옮기면 됩니다.

* 이를 공식으로 옮기면 다음과 같습니다.

* $x_i$+1 = $x_i$ −이동거리 $×$ 기울기의 부호

* 위의 식에서는 기울기의 부호는 알 수 있지만, 이동 거리는 어떻게 구해야할까요?

* 미분 계수(=기울기=gradient)는 극소값에 가까워질 수록 값이 작아집니다. 따라서 이동거리에는 미분 계수와 비례하는 값을 이용한다. 그럼 극소값에서 멀 때는 많이 이동하고 극소값에 가까울 때는 조금씩 이동할 수 있습니다.

* 이렇게해서 유도되는 최종 공식은 다음과 같습니다.

* $x_i+1$ = $x_i$ - $\alpha$ $dx \over df$ $(x_i)$

# Step Size(= $\alpha$ )

* 적절한 step size선택은 매우 중요합니다. Step size가 큰 경우 이동 거리가 커지므로 빠르게 수렴할 수 있다는 장점이 있지만 최소값으로 수렴되지 못하고 함수값이 발산할 여지가 있습니다.

* step size가 너무 작으면 발산하지는 않겠지만 최소값을 찾는데 너무 오래 걸릴 여지가 있으므로 적절한 step size를 찾아야합니다.

![image](https://user-images.githubusercontent.com/44185037/211530836-fbaa4966-84e6-4977-ae4e-182afea1b7fa.png)

출처 : https://angeloyeo.github.io/2020/08/16/gradient_descent.html

# Local minima

* Local minima 문제는 에러를 최소화시키는 최적의 파라미터를 찾는 문제에 있어서 아래 그림처럼 파라미터 공간에 수많은 지역적인 홀(hole)들이 존재하여 이러한 local minima에 빠질 경우 전역적인 해(global minimum)를 찾기 힘들게 되는 문제를 일컫습니다.

![image](https://user-images.githubusercontent.com/44185037/211531730-d59ed642-6cda-4b3b-82b2-a45e52e2cfc0.png)


![image](https://user-images.githubusercontent.com/44185037/211531151-c3f4bba4-0f1f-4667-9e49-23081595f9e5.png)

출처 : https://angeloyeo.github.io/2020/08/16/gradient_descent.html

* 하지만 최근에는 실제로 딥러닝이 수행될 때 local minima에 빠질 확률이 거의 없다고합니다.

* 위의 그래프에서는 가중치(w)가 1개인 모델이지만 실제 딥러닝 모델에서는 w가 셀 수없이 많아서 그 수많은 w가 모두 local minima에 빠져야 w 업데이트가 정지됩니다.

* 이론적으로는 거의 불가능에 가까운 일이므로, 사실상 local minima는 고려할 필요가 없는 것이 중론입니다.

* 해당 내용은 https://darkpgmr.tistory.com/148 이곳에 자세히 서술되어있습니다.

# 편미분 (partial differentiation)

* 여기서 하나의 개념을 더짚고 넘어가야합니다. 그것은 바로 **편미분(partial differentiation)**

* 변수가 벡터인 다변량 함수의 경우에는 편미분(partial differentiation)을 사용하여 기울기를 구해야합니다.

![image](https://user-images.githubusercontent.com/44185037/211511321-8797eb27-ee54-43e3-ab13-c0cc720f4b53.png)

마찬가지로 sympy library를 통해 구현한 편미분입니다.
```python
import sympy as sym 
from sympy.abc import x, y 

sym.diff(sym.poly(x**2 + 2*x*y + 3) + sym.cos(x + 2*y), x)
```

# gradient vector
* gradient vector는 변수 d개에 대해 각각 편미분 한 것을 한번에 표시합니다.

![image](https://user-images.githubusercontent.com/44185037/211511870-26267158-92e8-4f69-b6ef-cf458ca18d92.png)

위의 역삼각형 기호는 nabla라고 하며 gradient vector임을 나타내는 기호입니다. f`(x)대신 nabla를 사용하여 변수 x=(x1,x2,...,xd)를 동시에 update할 수 있습니다.
