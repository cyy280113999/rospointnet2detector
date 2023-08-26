import asyncio
import threading
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)
from pymodbus.server import (
    # ModbusTcpServer,
    ServerStop,
    StartAsyncTcpServer,
)
# from pymodbus.transaction import (
#     ModbusAsciiFramer,
#     ModbusBinaryFramer,
#     ModbusRtuFramer,
#     ModbusSocketFramer,
#     ModbusTlsFramer,
# )
import global_config as CONF


class MServer:
    def __init__(self):
        pass
    def create_server(self):
        datablock = ModbusSequentialDataBlock(0x00, [0] * 100)
        context = ModbusSlaveContext(
            di=datablock, co=datablock, hr=datablock, ir=datablock, zero_mode=True, # zero mode not work, default True
        )
        context = ModbusServerContext(slaves=context, single=True)
        print('mserver run')
        asyncio.run(StartAsyncTcpServer(
            context=context,  # Data storage
            identity=None,  # server identify
            address=(CONF.mip, CONF.mport),  # listen address
            framer=None,  # The framer strategy to use
            allow_reuse_address=True,  # allow the reuse of an address
            # timeout=1,  # waiting time for request to complete
        ))
        # self.server = ModbusTcpServer(
        #     context=context,  # Data storage
        #     identity=None,  # server identify
        #     address=(mip, mport),  # listen address
        #     framer=None,  # The framer strategy to use
        #     allow_reuse_address=True,  # allow the reuse of an address
        #     # timeout=1,  # waiting time for request to complete
        # )
        # self.server.serve_forever()
    def start(self):
        self.mserver_ptr = threading.Thread(target=self.create_server)
        self.mserver_ptr.start()
        # self.create_server()
    def stop(self):
        ServerStop()
        self.mserver_ptr.join()
        # self.server.shutdown()
        print('mserver shutdown')


if __name__=='__main__':
    print('start')
    s = MServer()
    s.start()
    while True:
        cmd=input()
        if cmd=='q':
            break
    s.stop()
