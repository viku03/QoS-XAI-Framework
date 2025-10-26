# Ryu SDN Controller Integration
"""
Ryu-based SDN controller that integrates with XAI-QoS framework
Handles real-time policy enforcement and traffic monitoring
"""

import logging
import threading
import time
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
from ryu.controller.controller import Datapath


from policy_translator import QoSPolicy, QoSPriority

class XAIQoSController(app_manager.RyuApp):
    """
    Ryu-based SDN controller that enforces QoS policies based on XAI explanations
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, config: Dict[str, Any], policy_translator, *args, **kwargs):
        super(XAIQoSController, self).__init__(*args, **kwargs)
        self.config = config
        self.policy_translator = policy_translator
        self.logger = logging.getLogger(__name__)
        
        # Network state
        self.datapaths = {}
        self.mac_to_port = {}
        
        # Traffic monitoring
        self.flow_stats = defaultdict(dict)
        self.active_flows = {}
        self.flow_counters = defaultdict(int)
        
        # QoS state
        self.active_policies = {}
        self.queue_configs = {}
        self.policy_stats = defaultdict(dict)
        
        # Monitoring configuration
        self.monitoring_interval = config.get('monitoring_interval', 5)  # seconds
        self.flow_timeout = config.get('flow_timeout', 300)  # seconds
        
        # Traffic collection for XAI processing
        self.traffic_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring_thread = None
        self.running = False
        
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        self.datapaths[datapath.id] = datapath
        self.logger.info(f"Switch {datapath.id} connected")
        
        # Install default flow (table-miss flow entry)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)
        
        # Setup QoS queues
        self._setup_queues(datapath)
        
    def _setup_queues(self, datapath):
        """Setup QoS queues on the switch"""
        try:
            # Create queue configurations for different priority levels
            queue_configs = {
                1: {'max_rate': 100000, 'min_rate': 50000},  # Critical
                2: {'max_rate': 50000, 'min_rate': 20000},   # High  
                3: {'max_rate': 20000, 'min_rate': 5000},    # Normal
                4: {'max_rate': 10000, 'min_rate': 1000},    # Low
                5: {'max_rate': 5000, 'min_rate': 500}       # Background
            }
            
            self.queue_configs[datapath.id] = queue_configs
            self.logger.info(f"Setup queues for switch {datapath.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup queues: {e}")
            
    def _add_flow(self, datapath, priority, match, actions, buffer_id=None,
                  idle_timeout=0, hard_timeout=0, queue_id=None):
        """Add a flow entry to the switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        if queue_id:
            actions.append(parser.OFPActionSetQueue(queue_id))
            
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match,
                                  instructions=inst, idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst,
                                  idle_timeout=idle_timeout,
                                  hard_timeout=hard_timeout)
            
        datapath.send_msg(mod)
        
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet-in events"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP packets
        if eth.ethertype == 0x88CC:
            return
            
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Learn MAC addresses
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        
        # Extract flow information for XAI processing
        flow_info = self._extract_flow_info(pkt, in_port, dpid)
        if flow_info:
            with self.buffer_lock:
                self.traffic_buffer.append(flow_info)
                
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
            
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Check if there's an active policy for this flow
            flow_id = self._generate_flow_id(flow_info) if flow_info else None
            if flow_id and flow_id in self.active_policies:
                policy = self.active_policies[flow_id]
                queue_id = policy.queue_id
                priority = 1000 + policy.priority.value  # Higher priority for QoS flows
            else:
                queue_id = None
                priority = 1
                
            self._add_flow(datapath, priority, match, actions,
                          queue_id=queue_id, idle_timeout=300)
                          
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
        
    def _extract_flow_info(self, pkt, in_port: int, dpid: int) -> Optional[Dict[str, Any]]:
        """Extract flow information from packet for XAI processing"""
        try:
            eth = pkt.get_protocol(ethernet.ethernet)
            if not eth:
                return None
                
            flow_info = {
                'timestamp': time.time(),
                'dpid': dpid,
                'in_port': in_port,
                'eth_src': eth.src,
                'eth_dst': eth.dst,
                'eth_type': eth.ethertype
            }
            
            # Extract IP information
            ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
            if ipv4_pkt:
                flow_info.update({
                    'src_ip': ipv4_pkt.src,
                    'dst_ip': ipv4_pkt.dst,
                    'protocol': ipv4_pkt.proto,
                    'packet_size': len(pkt.data)
                })
                
                # Extract transport layer information
                tcp_pkt = pkt.get_protocol(tcp.tcp)
                udp_pkt = pkt.get_protocol(udp.udp)
                
                if tcp_pkt:
                    flow_info.update({
                        'src_port': tcp_pkt.src_port,
                        'dst_port': tcp_pkt.dst_port,
                        'tcp_flags': {
                            'syn': bool(tcp_pkt.bits & tcp.TCP_SYN),
                            'ack': bool(tcp_pkt.bits & tcp.TCP_ACK),
                            'fin': bool(tcp_pkt.bits & tcp.TCP_FIN),
                            'rst': bool(tcp_pkt.bits & tcp.TCP_RST),
                            'psh': bool(tcp_pkt.bits & tcp.TCP_PSH)
                        }
                    })
                elif udp_pkt:
                    flow_info.update({
                        'src_port': udp_pkt.src_port,
                        'dst_port': udp_pkt.dst_port
                    })
                    
            return flow_info
            
        except Exception as e:
            self.logger.error(f"Error extracting flow info: {e}")
            return None
            
    def _generate_flow_id(self, flow_info: Dict[str, Any]) -> str:
        """Generate unique flow identifier"""
        return f"{flow_info.get('src_ip', '')}-{flow_info.get('dst_ip', '')}-" \
               f"{flow_info.get('src_port', 0)}-{flow_info.get('dst_port', 0)}-" \
               f"{flow_info.get('protocol', 0)}"
               
    def start(self):
        """Start the controller and monitoring"""
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = hub.spawn(self._monitoring_loop)
        
        self.logger.info("XAI-QoS Controller started")
        
    def stop(self):
        """Stop the controller"""
        self.running = False
        
        if self.monitoring_thread:
            hub.kill(self.monitoring_thread)
            
        self.logger.info("XAI-QoS Controller stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Request flow statistics from all switches
                for dpid, datapath in self.datapaths.items():
                    self._request_flow_stats(datapath)
                    
                # Clean up old flows
                self._cleanup_old_flows()
                
                hub.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                hub.sleep(1)
                
    def _request_flow_stats(self, datapath):
        """Request flow statistics from switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply"""
        flows = []
        for stat in ev.msg.body:
            flows.append({
                'table_id': stat.table_id,
                'match': stat.match,
                'priority': stat.priority,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec
            })
            
        self.flow_stats[ev.msg.datapath.id] = flows
        
    def get_current_traffic(self) -> List[Dict[str, Any]]:
        """Get current traffic data for XAI processing"""
        with self.buffer_lock:
            traffic_data = list(self.traffic_buffer)
            self.traffic_buffer.clear()
            
        # Aggregate traffic data into flows
        flows = self._aggregate_traffic_to_flows(traffic_data)
        return flows
        
    def _aggregate_traffic_to_flows(self, traffic_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate packet data into flow statistics"""
        flow_aggregates = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'first_seen': float('inf'),
            'last_seen': 0,
            'packet_sizes': [],
            'inter_arrival_times': [],
            'tcp_flags': defaultdict(int)
        })
        
        # Process each packet
        for pkt_info in traffic_data:
            if 'src_ip' not in pkt_info:
                continue
                
            flow_id = self._generate_flow_id(pkt_info)
            flow_data = flow_aggregates[flow_id]
            
            # Update counters
            flow_data['packet_count'] += 1
            flow_data['byte_count'] += pkt_info.get('packet_size', 0)
            
            # Update timing
            timestamp = pkt_info['timestamp']
            flow_data['first_seen'] = min(flow_data['first_seen'], timestamp)
            flow_data['last_seen'] = max(flow_data['last_seen'], timestamp)
            
            # Collect packet sizes
            flow_data['packet_sizes'].append(pkt_info.get('packet_size', 0))
            
            # Calculate inter-arrival times
            if len(flow_data['packet_sizes']) > 1:
                last_timestamp = getattr(flow_data, 'prev_timestamp', timestamp)
                iat = timestamp - last_timestamp
                flow_data['inter_arrival_times'].append(iat)
            flow_data['prev_timestamp'] = timestamp
            
            # TCP flags
            tcp_flags = pkt_info.get('tcp_flags', {})
            for flag, value in tcp_flags.items():
                if value:
                    flow_data['tcp_flags'][flag] += 1
                    
            # Copy other fields
            for field in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']:
                if field in pkt_info:
                    flow_data[field] = pkt_info[field]
                    
        # Convert to final format
        flows = []
        for flow_id, data in flow_aggregates.items():
            if data['packet_count'] == 0:
                continue
                
            # Calculate derived statistics
            duration = max(data['last_seen'] - data['first_seen'], 0.001)
            
            flow = {
                'flow_id': flow_id,
                'packet_count': data['packet_count'],
                'byte_count': data['byte_count'],
                'duration': duration,
                'packet_rate': data['packet_count'] / duration,
                'byte_rate': data['byte_count'] / duration,
                'avg_packet_size': data['byte_count'] / data['packet_count'] if data['packet_count'] > 0 else 0,
                'packet_sizes': data['packet_sizes'],
                'inter_arrival_times': data['inter_arrival_times'],
                'tcp_flags': dict(data['tcp_flags']),
                'src_ip': data.get('src_ip', ''),
                'dst_ip': data.get('dst_ip', ''),
                'src_port': data.get('src_port', 0),
                'dst_port': data.get('dst_port', 0),
                'protocol': data.get('protocol', 0),
                'timestamp': data['first_seen']
            }
            
            flows.append(flow)
            
        return flows
        
    def enforce_policies(self, policies: List[QoSPolicy], flow_data: Dict[str, Any]):
        """Enforce QoS policies on the network"""
        try:
            for policy in policies:
                flow_id = policy.flow_id
                
                # Store active policy
                self.active_policies[flow_id] = policy
                
                # Apply policy to all relevant switches
                for dpid, datapath in self.datapaths.items():
                    self._apply_policy_to_switch(datapath, policy, flow_data)
                    
                self.logger.info(f"Enforced policy for flow {flow_id}: {policy.priority.name} priority")
                
        except Exception as e:
            self.logger.error(f"Error enforcing policies: {e}")
            
    def _apply_policy_to_switch(self, datapath, policy: QoSPolicy, flow_data: Dict[str, Any]):
        """Apply QoS policy to specific switch"""
        try:
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            
            # Create match for the flow
            match_kwargs = {}
            
            if flow_data.get('src_ip'):
                match_kwargs['ipv4_src'] = flow_data['src_ip']
            if flow_data.get('dst_ip'):
                match_kwargs['ipv4_dst'] = flow_data['dst_ip']
            if flow_data.get('src_port'):
                match_kwargs['tcp_src'] = flow_data['src_port']
            if flow_data.get('dst_port'):
                match_kwargs['tcp_dst'] = flow_data['dst_port']
                
            match = parser.OFPMatch(**match_kwargs)
            
            # Create actions based on policy
            actions = []
            
            # Set queue for QoS
            actions.append(parser.OFPActionSetQueue(policy.queue_id))
            
            # Set DSCP marking if specified
            if policy.dscp_marking > 0:
                actions.append(parser.OFPActionSetField(ip_dscp=policy.dscp_marking))
                
            # Forward to appropriate port (simplified - would need proper routing)
            actions.append(parser.OFPActionOutput(ofproto.OFPP_NORMAL))
            
            # Calculate priority based on policy priority
            priority = 1000 + (6 - policy.priority.value) * 100
            
            # Add flow with policy
            self._add_flow(
                datapath=datapath,
                priority=priority,
                match=match,
                actions=actions,
                idle_timeout=300,
                hard_timeout=600,
                queue_id=policy.queue_id
            )
            
        except Exception as e:
            self.logger.error(f"Error applying policy to switch {datapath.id}: {e}")
            
    def _cleanup_old_flows(self):
        """Remove old inactive flows from tracking"""
        current_time = time.time()
        expired_flows = []
        
        for flow_id, timestamp in list(self.active_flows.items()):
            if current_time - timestamp > self.flow_timeout:
                expired_flows.append(flow_id)
                
        for flow_id in expired_flows:
            del self.active_flows[flow_id]
            if flow_id in self.active_policies:
                del self.active_policies[flow_id]
                
        if expired_flows:
            self.logger.debug(f"Cleaned up {len(expired_flows)} expired flows")
            
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about active policies"""
        return {
            'active_policies': len(self.active_policies),
            'switches_connected': len(self.datapaths),
            'flows_tracked': len(self.active_flows),
            'policy_distribution': {
                priority.name: sum(1 for p in self.active_policies.values() 
                                 if p.priority == priority)
                for priority in QoSPriority
            }
        }