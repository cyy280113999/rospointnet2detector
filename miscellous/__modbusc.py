#!/usr/bin/env python3
"""Pymodbus Client modbus call examples.

Please see:

    async_template_call

    template_call

for a template on how to make modbus calls and check for different
error conditions.

The _handle_.... functions each handle a set of modbus calls with the
same register type (e.g. coils).

All available modbus calls are present. The difference between async
and sync is a single 'await' so the calls are not repeated.

If you are performing a request that is not available in the client
mixin, you have to perform the request like this instead:

from pymodbus.diag_message import ClearCountersRequest
from pymodbus.diag_message import ClearCountersResponse

request  = ClearCountersRequest()
response = client.execute(request)
if isinstance(response, ClearCountersResponse):
    ... do something with the response

This example uses client_async.py and client_sync.py to handle connection,
and have the same options.

The corresponding server must be started before e.g. as:

    ./server_async.py
"""
import asyncio
import logging

import pymodbus.diag_message as req_diag
import pymodbus.mei_message as req_mei
import pymodbus.other_message as req_other
from pymodbus.exceptions import ModbusException
from pymodbus.pdu import ExceptionResponse
import asyncio
import logging
from pymodbus.client import AsyncModbusTcpClient,ModbusTcpClient
from modbuss import get_commandline


SLAVE = 0x00


_logger = logging.getLogger()
_logger.setLevel("DEBUG")


def setup_sync_client(description=None, cmdline=None):
    """Run client setup."""
    _logger.info("### Create client object")
    if args.comm == "tcp":
        client = ModbusTcpClient(
            args.host,
            port=args.port,
            # Common optional paramers:
            framer=args.framer,
            #    timeout=10,
            #    retries=3,
            #    retry_on_empty=False,y
            #    close_comm_on_error=False,
            #    strict=True,
            # TCP setup parameters
            #    source_address=("localhost", 0),
        )
    return client
def run_sync_client(client, modbus_calls=None):
    """Run sync client."""
    _logger.info("### Client starting")
    client.connect()
    if modbus_calls:
        modbus_calls(client)
    client.close()
    _logger.info("### End of Program")

def setup_async_client(description=None, cmdline=None):
    """Run client setup."""
    args = get_commandline(
        server=False, description=description, cmdline=cmdline
    )
    args.port = port
    _logger.info("### Create client object")
    if args.comm == "tcp":
        client = AsyncModbusTcpClient(
            args.host,
            port=args.port,  # on which port
            # Common optional paramers:
            framer=args.framer,
            #    timeout=10,
            #    retries=3,
            #    retry_on_empty=False,
            #    close_comm_on_error=False,
            #    strict=True,
            # TCP setup parameters
            #    source_address=("localhost", 0),
        )
    return client
async def run_async_client(client, modbus_calls=None):
    """Run sync client."""
    _logger.info("### Client starting")
    await client.connect()
    assert client.connected
    if modbus_calls:
        await modbus_calls(client)
    client.close()
    _logger.info("### End of Program")


# ------- caller

async def async_template_call(client):
    """Show complete modbus call, async version."""
    try:
        rr = await client.read_coils(1, 1, slave=SLAVE)
    except ModbusException as exc:
        txt = f"ERROR: exception in pymodbus {exc}"
        _logger.error(txt)
        raise exc
    if rr.isError():
        txt = "ERROR: pymodbus returned an error!"
        _logger.error(txt)
        raise ModbusException(txt)
    if isinstance(rr, ExceptionResponse):
        txt = "ERROR: received exception from device {rr}!"
        _logger.error(txt)
        # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
        raise ModbusException(txt)

    # Validate data
    txt = f"### Template coils response: {str(rr.bits)}"
    _logger.debug(txt)
# -------------------------------------------------
# Generic error handling, to avoid duplicating code
# -------------------------------------------------
def _check_call(rr):
    """Check modbus call worked generically."""
    assert not rr.isError()  # test that call was OK
    assert not isinstance(rr, ExceptionResponse)  # Device rejected request
    return rr



async def _handle_coils(client):
    """Read/Write coils."""
    _logger.info("### Reading Coil different number of bits (return 8 bits multiples)")
    rr = _check_call(await client.read_coils(1, 1, slave=SLAVE))
    assert len(rr.bits) == 8

    rr = _check_call(await client.read_coils(1, 5, slave=SLAVE))
    assert len(rr.bits) == 8

    rr = _check_call(await client.read_coils(1, 12, slave=SLAVE))
    assert len(rr.bits) == 16

    rr = _check_call(await client.read_coils(1, 17, slave=SLAVE))
    assert len(rr.bits) == 24

    _logger.info("### Write false/true to coils and read to verify")
    _check_call(await client.write_coil(0, True, slave=SLAVE))
    rr = _check_call(await client.read_coils(0, 1, slave=SLAVE))
    assert rr.bits[0]  # test the expected value

    _check_call(await client.write_coils(1, [True] * 21, slave=SLAVE))
    rr = _check_call(await client.read_coils(1, 21, slave=SLAVE))
    resp = [True] * 21
    # If the returned output quantity is not a multiple of eight,
    # the remaining bits in the final data byte will be padded with zeros
    # (toward the high order end of the byte).
    resp.extend([False] * 3)
    assert rr.bits == resp  # test the expected value

    _logger.info("### Write False to address 1-8 coils")
    _check_call(await client.write_coils(1, [False] * 8, slave=SLAVE))
    rr = _check_call(await client.read_coils(1, 8, slave=SLAVE))
    assert rr.bits == [False] * 8  # test the expected value


async def _handle_discrete_input(client):
    """Read discrete inputs."""
    _logger.info("### Reading discrete input, Read address:0-7")
    rr = _check_call(await client.read_discrete_inputs(0, 8, slave=SLAVE))
    assert len(rr.bits) == 8


async def _handle_holding_registers(client):
    """Read/write holding registers."""
    _logger.info("### write holding register and read holding registers")
    _check_call(await client.write_register(1, 10, slave=SLAVE))
    rr = _check_call(await client.read_holding_registers(1, 1, slave=SLAVE))
    assert rr.registers[0] == 10

    _check_call(await client.write_registers(1, [10] * 8, slave=SLAVE))
    rr = _check_call(await client.read_holding_registers(1, 8, slave=SLAVE))
    assert rr.registers == [10] * 8

    _logger.info("### write read holding registers, using **kwargs")
    arguments = {
        "read_address": 1,
        "read_count": 8,
        "write_address": 1,
        "values": [256, 128, 100, 50, 25, 10, 5, 1],
    }
    _check_call(await client.readwrite_registers(slave=SLAVE, **arguments))
    rr = _check_call(await client.read_holding_registers(1, 8, slave=SLAVE))
    assert rr.registers == arguments["values"]


async def _handle_input_registers(client):
    """Read input registers."""
    _logger.info("### read input registers")
    rr = _check_call(await client.read_input_registers(1, 8, slave=SLAVE))
    assert len(rr.registers) == 8


async def _execute_information_requests(client):
    """Execute extended information requests."""
    _logger.info("### Running information requests.")
    rr = _check_call(
        await client.execute(req_mei.ReadDeviceInformationRequest(slave=SLAVE))
    )
    assert rr.information[0] == b"Pymodbus"

    rr = _check_call(await client.execute(req_other.ReportSlaveIdRequest(slave=SLAVE)))
    assert rr.status

    rr = _check_call(
        await client.execute(req_other.ReadExceptionStatusRequest(slave=SLAVE))
    )
    assert not rr.status

    rr = _check_call(
        await client.execute(req_other.GetCommEventCounterRequest(slave=SLAVE))
    )
    assert rr.status
    assert not rr.count

    rr = _check_call(
        await client.execute(req_other.GetCommEventLogRequest(slave=SLAVE))
    )
    assert rr.status
    assert not (rr.event_count + rr.message_count + len(rr.events))


async def _execute_diagnostic_requests(client):
    """Execute extended diagnostic requests."""
    _logger.info("### Running diagnostic requests.")
    rr = _check_call(await client.execute(req_diag.ReturnQueryDataRequest(slave=SLAVE)))
    assert not rr.message[0]

    _check_call(
        await client.execute(req_diag.RestartCommunicationsOptionRequest(slave=SLAVE))
    )
    _check_call(
        await client.execute(req_diag.ReturnDiagnosticRegisterRequest(slave=SLAVE))
    )
    _check_call(
        await client.execute(req_diag.ChangeAsciiInputDelimiterRequest(slave=SLAVE))
    )

    # NOT WORKING: _check_call(await client.execute(req_diag.ForceListenOnlyModeRequest(slave=SLAVE)))
    # does not send a response

    _check_call(await client.execute(req_diag.ClearCountersRequest()))
    _check_call(
        await client.execute(
            req_diag.ReturnBusCommunicationErrorCountRequest(slave=SLAVE)
        )
    )
    _check_call(
        await client.execute(req_diag.ReturnBusExceptionErrorCountRequest(slave=SLAVE))
    )
    _check_call(
        await client.execute(req_diag.ReturnSlaveMessageCountRequest(slave=SLAVE))
    )
    _check_call(
        await client.execute(req_diag.ReturnSlaveNoResponseCountRequest(slave=SLAVE))
    )
    _check_call(await client.execute(req_diag.ReturnSlaveNAKCountRequest(slave=SLAVE)))
    _check_call(await client.execute(req_diag.ReturnSlaveBusyCountRequest(slave=SLAVE)))
    _check_call(
        await client.execute(
            req_diag.ReturnSlaveBusCharacterOverrunCountRequest(slave=SLAVE)
        )
    )
    _check_call(
        await client.execute(req_diag.ReturnIopOverrunCountRequest(slave=SLAVE))
    )
    _check_call(await client.execute(req_diag.ClearOverrunCountRequest(slave=SLAVE)))
    # NOT WORKING _check_call(await client.execute(req_diag.GetClearModbusPlusRequest(slave=SLAVE)))


# def int32toQWord(x:int):
#     # return list of double DWord
#     value_bytes = x.to_bytes(4, byteorder='big', signed=True)
#     hd:bytes = value_bytes[:2]
#     ld = value_bytes[2:]
#     hd = int(hd)
#     hd = hex(hd)
def encode_value(value):# int32 to DWord * 2
    high_word = (value >> 16) & 0xFFFF  # 取高16位
    low_word = value & 0xFFFF  # 取低16位
    return high_word, low_word
def decode_value(data):
    high_word, low_word = data
    value = (high_word << 16) | low_word  # 合并高16位和低16位
    if value & 0x80000000:  # 检查符号位
        value -= 0x100000000  # 转换为有符号整数
    return value

def sync_main(client):
    """Test connection works."""
    # rr = client.read_coils(32, 1, slave=1)
    # assert len(rr.bits) == 8
    # rr = client.read_holding_registers(4, 2, slave=1)
    # assert rr.registers[0] == 17
    # assert rr.registers[1] == 17

    #     arguments = {
    #     "read_address": 1,
    #     "read_count": 8,
    #     "write_address": 1,
    #     "values": [256, 128, 100, 50, 25, 10, 5, 1],
    # }
    # _check_call(await client.readwrite_registers(slave=SLAVE, **arguments))
    # rr = _check_call(await client.read_holding_registers(1, 8, slave=SLAVE))
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
            # value_bytes = value.to_bytes(4, byteorder='big', signed=True)
              # 参数说明：address为寄存器地址，value为写入的值，unit为设备地址
            result = client.write_registers(addr,value,unit=SLAVE)
            if result.isError():
                print("写入错误：", result)
            else:
                print("写入成功")
        elif cmd[0]=='r' or cmd[0]=='read':
            if len(cmd)==1:
                continue
            addr=int(cmd[1])
            count=2
            if len(cmd)==3:
                count=int(cmd[2])
            # 参数说明：address为寄存器地址，count为读取的寄存器数量，unit为设备地址
            result = client.read_holding_registers(address=addr, count=count, unit=SLAVE)  
            if result.isError():
                print("读取错误：", result)
            else:
                print("读取的值：", result.registers)
        elif cmd[0]=='wq':
            if len(cmd)!=3:
                continue
            addr=int(cmd[1])
            value = int(cmd[2]) # write qword
            result = client.write_registers(addr,encode_value(value),unit=SLAVE) # write at 10
            if result.isError():
                print("写入错误：", result)
            else:
                print("写入成功")
        elif cmd[0]=='rq':
            if len(cmd)==1:
                continue
            addr=int(cmd[1])
            # 参数说明：address为寄存器地址，count为读取的寄存器数量，unit为设备地址
            result = client.read_holding_registers(address=addr, count=2, unit=SLAVE)  
            if result.isError():
                print("读取错误：", result)
            else:
                print("读取的值：", decode_value(result.registers))
def run_sync_calls(client):
    """Demonstrate basic read/write calls."""
    """Show complete modbus call, sync version."""
    try:
        rr = client.read_coils(1, 1, slave=SLAVE)
    except ModbusException as exc:
        txt = f"ERROR: exception in pymodbus {exc}"
        _logger.error(txt)
        raise exc
    if rr.isError():
        txt = "ERROR: pymodbus returned an error!"
        _logger.error(txt)
        raise ModbusException(txt)
    if isinstance(rr, ExceptionResponse):
        txt = "ERROR: received exception from device {rr}!"
        _logger.error(txt)
        # THIS IS NOT A PYTHON EXCEPTION, but a valid modbus message
        raise ModbusException(txt)

    # Validate data
    txt = f"### Template coils response: {str(rr.bits)}"
    _logger.debug(txt)
async def run_async_calls(client):
    """Demonstrate basic read/write calls."""
    # await async_template_call(client)
    # await _handle_coils(client)
    # await _handle_discrete_input(client)
    await _handle_holding_registers(client)
    # await _handle_input_registers(client)
    # await _execute_information_requests(client)
    # await _execute_diagnostic_requests(client)
async def async_main(): 
    """Combine the setup and run"""
    testclient = setup_async_client(description="Run asynchronous client.")
    await run_async_client(testclient, modbus_calls=run_async_calls)


if __name__ == "__main__":
    global args
    args = get_commandline(
        server=False,
        description="Run client.",
        cmdline=None,
    )
    # use global settings
    mip='192.168.10.195'
    mport=12340
    args.host='192.168.10.195'
    args.port = 12340
    testclient = setup_sync_client()
    run_sync_client(testclient, modbus_calls=sync_main)
    # asyncio.run(async_main())
    # testclient = setup_sync_client(
    #     description="Run modbus calls in synchronous client."
    # )
    # run_sync_client(testclient, modbus_calls=run_sync_calls)