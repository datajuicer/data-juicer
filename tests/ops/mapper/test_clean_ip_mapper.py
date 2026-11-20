import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_ip_mapper import CleanIpMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CleanIpMapperTest(DataJuicerTestCaseBase):

    def _run_clean_ip(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_ipv4(self):

        samples = [{
            'text': 'test of ip 234.128.124.123',
            'target': 'test of ip '
        }, {
            'text': '34.0.124.123',
            'target': ''
        }, {
            'text': 'ftp://example.com/188.46.244.216my-page.html',
            'target': 'ftp://example.com/my-page.html'
        }, {
            'text': 'ft174.1421.237.246my',
            'target': 'ft174.1421.237.246my'
        }]
        op = CleanIpMapper()
        self._run_clean_ip(op, samples)

    def test_ipv6(self):

        samples = [{
            'text': 'dd41:cbaf:d1b4:10a0:b215:72e3:6eaf:3ecb',
            'target': ''
        }, {
            'text': 'test of ip 4394:538a:3bf3:61c3:cb0d:d214:526f:70d',
            'target': 'test of ip '
        }, {
            'text': 'com/f770:c52e:ddce:3a9f:8c3b:a7bd:d81f:985cmy-page.html',
            'target': 'com/my-page.html'
        }, {
            'text': 'ft1926:43a1:fcb5:ees06:ae63:a2a4:c656:d014my',
            'target': 'ft1926:43a1:fcb5:ees06:ae63:a2a4:c656:d014my'
        }]
        op = CleanIpMapper()
        self._run_clean_ip(op, samples)

    def test_compressed_ipv6(self):

        samples = [{
            'text': 'test of ip 2001:db8::ff00:42:8329',
            'target': 'test of ip '
        }, {
            'text': 'loopback ::1 and link-local FE80::1FF:FE23:4567:890A',
            'target': 'loopback  and link-local '
        }, {
            'text': 'connect to [2001:db8::1]:443 now',
            'target': 'connect to []:443 now'
        }, {
            'text': '地址2001:db8::1，请勿外传',
            'target': '地址，请勿外传'
        }, {
            'text': 'std::cout at 12:30 from 00:1B:44:11:3A:B7',
            'target': 'std::cout at 12:30 from 00:1B:44:11:3A:B7'
        }]
        op = CleanIpMapper()
        self._run_clean_ip(op, samples)

    def test_ipv4_embedded_ipv6(self):

        samples = [{
            'text': 'mapped ::ffff:192.0.2.128 ok',
            'target': 'mapped  ok'
        }, {
            'text': 'connect to [::ffff:192.0.2.128]:443 now',
            'target': 'connect to []:443 now'
        }, {
            'text': 'nat64 64:ff9b::198.51.100.7, full 0:0:0:0:0:ffff:192.0.2.1.',
            'target': 'nat64 , full .'
        }]
        op = CleanIpMapper()
        self._run_clean_ip(op, samples)

    def test_replace_ipv4_embedded_ipv6(self):

        samples = [{
            'text': '::ffff:192.0.2.128',
            'target': '<IP>'
        }, {
            'text': '[::ffff:192.0.2.128]:443',
            'target': '[<IP>]:443'
        }]
        op = CleanIpMapper(repl='<IP>')
        self._run_clean_ip(op, samples)

    def test_replace_ipv4(self):

        samples = [{
            'text': 'test of ip 234.128.124.123',
            'target': 'test of ip <IP>'
        }, {
            'text': '34.0.124.123',
            'target': '<IP>'
        }, {
            'text': 'ftp://example.com/188.46.244.216my-page.html',
            'target': 'ftp://example.com/<IP>my-page.html'
        }, {
            'text': 'ft174.1421.237.246my',
            'target': 'ft174.1421.237.246my'
        }]
        op = CleanIpMapper(repl='<IP>')
        self._run_clean_ip(op, samples)


if __name__ == '__main__':
    unittest.main()
