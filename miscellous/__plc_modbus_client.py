from pymodbus.client.tcp import AsyncModbusTcpClient

def main():
    ip = '127.0.0.1'
    # ip = '192.168.1.128'
    # ip = '192.168.10.195'
    port = 12340
    device_id=255
    # ip = '192.168.10.1'
    # port=2000
    # device_id=5
    # 创建Modbus TCP客户端
    client = AsyncModbusTcpClient(ip, port=port)  # 替换为PLC的IP地址

    # 连接到PLC
    client.connect()

    # 读取保持寄存器的值
    while True:
        print('input command:')
        cmd = input().split()
        if len(cmd)==0:
            continue
        if cmd[0]=='q'or cmd[0]=='quit':
            break
        elif cmd[0]=='w' or cmd[0]=='write':
            if len(cmd)!=3:
                continue
            addr=int(cmd[1])
            value = int(cmd[2])
              # 参数说明：address为寄存器地址，value为写入的值，unit为设备地址
            result = client.write_register(addr,value,unit=device_id)
            if result.isError():
                print("写入错误：", result)
            else:
                print("写入成功")
        elif cmd[0]=='r' or cmd[0]=='read':
            if len(cmd)!=3:
                continue
            addr=int(cmd[1])
            count=int(cmd[2])
            # 参数说明：address为寄存器地址，count为读取的寄存器数量，unit为设备地址
            result = client.read_holding_registers(address=addr, count=count, unit=device_id)  
            if result.isError():
                print("读取错误：", result)
            else:
                print("读取的值：", result.registers)

    # 断开与PLC的连接
    client.close()

if __name__ == '__main__':
    main()