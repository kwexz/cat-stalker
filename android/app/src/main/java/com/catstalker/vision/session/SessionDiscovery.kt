package com.catstalker.vision.session

import android.content.Context
import android.net.wifi.WifiManager
import android.util.Log
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

/** Zero-touch PC discovery: broadcast, take the first collector reply. */
object SessionDiscovery {
    private const val TAG = "CatStalkerSession"
    private const val PORT = 8766
    private const val DISCOVER = "catstalker-discover-v1"
    private const val REPLY_PREFIX = "catstalker-collector-v1 "

    data class Found(val host: String, val port: Int)

    fun discover(context: Context, timeoutMs: Int = 2500): Found? {
        val targets = mutableListOf<InetAddress>()
        runCatching {
            val wifi = context.applicationContext
                .getSystemService(Context.WIFI_SERVICE) as? WifiManager
            @Suppress("DEPRECATION")
            val dhcp = wifi?.dhcpInfo
            if (wifi != null && dhcp != null && dhcp.ipAddress != 0) {
                val broadcast = (dhcp.ipAddress and dhcp.netmask) or dhcp.netmask.inv()
                val bytes = byteArrayOf(
                    (broadcast and 0xff).toByte(),
                    ((broadcast shr 8) and 0xff).toByte(),
                    ((broadcast shr 16) and 0xff).toByte(),
                    ((broadcast shr 24) and 0xff).toByte(),
                )
                targets.add(InetAddress.getByAddress(bytes))
            }
        }
        runCatching { targets.add(InetAddress.getByName("255.255.255.255")) }
        var socket: DatagramSocket? = null
        try {
            socket = DatagramSocket().apply {
                broadcast = true
                soTimeout = timeoutMs
            }
            val payload = DISCOVER.toByteArray(Charsets.US_ASCII)
            for (target in targets) {
                runCatching {
                    socket.send(DatagramPacket(payload, payload.size, target, PORT))
                }
            }
            val buf = ByteArray(64)
            val packet = DatagramPacket(buf, buf.size)
            val deadline = System.currentTimeMillis() + timeoutMs
            while (System.currentTimeMillis() < deadline) {
                try {
                    socket.receive(packet)
                } catch (e: java.net.SocketTimeoutException) {
                    break
                }
                val text = String(packet.data, 0, packet.length, Charsets.US_ASCII)
                if (text.startsWith(REPLY_PREFIX)) {
                    val port = text.removePrefix(REPLY_PREFIX).trim().toIntOrNull() ?: continue
                    return Found(packet.address.hostAddress ?: continue, port)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "discovery failed", e)
        } finally {
            socket?.close()
        }
        return null
    }
}
