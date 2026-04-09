from src.pemc import fredholm_pemc
import time

if __name__ == "__main__":
    LbyReff = 4.198707084443910054e-01
    theta1 = 0
    theta2 = 0
    u = 1/4
    LbyLambT = 0.1
    freq = 0.001

    num_spreng = -1.875317044961966884e-01-9.204718841737026536e-02

    start = time.time()

    logdet = fredholm_pemc(LbyReff, LbyLambT, theta1=theta1, theta2=theta2,
                           u=u, freq=freq, eta=6)

    print('run-time:', time.time()-start)
    print('freq=0.001:', '%.6f' % logdet[0])
    print('spreng:', '%.6f' % num_spreng)
