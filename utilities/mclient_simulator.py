from mclient import MClient
import time
import numpy as np
if __name__=='__main__':
    client=MClient()
    while True:
        data=client.get_require()
        if data[0]==1:
            print('got require')
            client.set_require(2)
            # ps=[[-1,-1,-1]]*13
            ps = np.random.randint(-200,-100,size=(13,3))
            for i,p in enumerate(ps):
                client.setpoint(i,p)
            client.set_require(3)
        time.sleep(0.1)
