# import asyncio
# import logging
from pymodbus.client import AsyncModbusTcpClient,ModbusTcpClient
from pymodbus.client import ModbusTcpClient
import numpy as np
from functools import reduce
import global_config as CONF

"""
modbus client

use read & write method to access continuous memory in server.

server register buffer: [uint16,]

write:
write_registers(start,data:list)

read:
read_holding_registers(start,len)

start:0
end:9999
data:[uint16,]

"""

def encode_value(value):# int32 to Word * 2
    high_word = (value >> 16) & 0xFFFF  # 取高16位
    low_word = value & 0xFFFF  # 取低16位
    return high_word, low_word
def decode_value(data):
    high_word, low_word = data
    value = (high_word << 16) | low_word  # 合并高16位和低16位
    if value & 0x80000000:  # 检查符号位
        value -= 0x100000000  # 转换为有符号整数
    return value

class MClient:
    def __init__(self):
        self.client = ModbusTcpClient(
            host=CONF.MODBUS_IP,
            port=CONF.MODBUS_PORT,
            # Common optional paramers:
            # framer=None,# framer must not None, default is socket framer
            #    timeout=10,
            #    retries=3,
            #    retry_on_empty=False,y
            #    close_comm_on_error=False,
            #    strict=True,
            # TCP setup parameters
            #    source_address=("localhost", 0),
        )
    def get_require(self):
        result = self.client.read_holding_registers(CONF.MODBUS_WORD_REQUIRE,1)
        ans = result.registers[0] # get first word, as uint16
        if result.isError():
            print("读取错误：", result)
            ans = None
        return ans
    def parse_require(self,x):
        req=0
        n=0
        if x in [CONF.REQUIRE_201,CONF.REQUIRE_202,CONF.REQUIRE_203]:
            req=1
            n=2
        elif x in [CONF.REQUIRE_13]:
            req=1
            n=13
        return req,n
    def set_require(self, state=3):
        result = self.client.write_registers(CONF.MODBUS_WORD_REQUIRE,[state])
        if result.isError():
            print("写入错误：", result)
    def set_moving(self, flag=True):
        result = self.client.write_registers(CONF.MODBUS_WORD_MOVE,[1 if flag else 0])
        if result.isError():
            print("写入错误：", result)
    def setpoint(self,n=0,p=[0,0,0]): # n < 13
        if isinstance(p, np.ndarray):
            p = p.tolist()
        data=reduce(lambda x,y:x+y,list(map(encode_value,p)))
        result = self.client.write_registers(CONF.MODBUS_WORD_POINTS+n*6,data) # write at 10
        if result.isError():
            print("写入错误：", result)
    def getpoint(self,n=0):
        result = self.client.read_holding_registers(CONF.MODBUS_WORD_POINTS+n*6,6)
        if result.isError():
            print("读取错误：", result)
        else:
            data=[]
            for i in range(3):
                data.append(decode_value(result.registers[i*2:i*2+2]))
            print('读取：', data)
    def start(self):
        self.client.connect()
    def stop(self):
        self.client.close()

if __name__=='__main__':
    client=MClient()
    while True:
        print('input command:')
        cmd = input()
        spliter=None
        if ',' in cmd:
            spliter=','
        elif ';' in cmd:
            spliter=';'
        cmd= cmd.split(spliter)
        if len(cmd)==0:
            continue
        if cmd[0]=='q':
            break
        elif cmd[0]=='w':
            if len(cmd)!=3:
                continue
            addr=int(cmd[1])
            value = int(cmd[2])
            result = client.client.write_registers(addr,value)
            if result.isError():
                print("写入错误：", result)
            else:
                print("写入成功")
        elif cmd[0]=='r':
            if len(cmd)==1:
                continue
            addr=int(cmd[1])
            count=2
            if len(cmd)==3:
                count=int(cmd[2])
            # 参数说明：address为寄存器地址，count为读取的寄存器数量，unit为设备地址
            result = client.client.read_holding_registers(address=addr, count=count)  
            if result.isError():
                print("读取错误：", result)
            else:
                print("读取的值：", result.registers)
        elif cmd[0]=='wq':
            if len(cmd)!=5:
                continue
            data=[0]*4
            try:
                for i in range(4):
                    data[i]=int(cmd[i+1])
            except:
                []
            client.setpoint(data[0],data[1:])
        elif cmd[0]=='rq':
            if len(cmd)==1:
                continue
            addr=int(cmd[1])
            client.getpoint(addr)  
        elif cmd[0]=='get':
            print(client.get_require())
            
