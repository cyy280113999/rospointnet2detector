'''
 * @Author: liuzhao 
 * @Last Modified time: 2022-10-05 09:56:13 
'''

from pymodbus.server.async_io import (
    StartTcpServer,
)
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
)

datablock = ModbusSequentialDataBlock.create()
context = ModbusSlaveContext(
    di=datablock,
    co=datablock,
    hr=datablock,
    ir=datablock,
    zero_mode=True,
    )
single = True

# Build data storage
store = ModbusServerContext(slaves=context, single=single)


if __name__ == '__main__':
    ip = '127.0.0.1'
    # ip = '192.168.1.128'
    # ip = '192.168.10.195'
    port = 12340
    address = (ip, port)
    StartTcpServer(
	    context=store,  # Data storage
	    address=address,  # listen address
	  	allow_reuse_address=True,  # allow the reuse of an address
	)

