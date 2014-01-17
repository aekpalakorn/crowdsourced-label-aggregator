#!/usr/bin/python 
""" Start a HIT server

Command-line args:

"""
import requests
import os.path
import simplejson
import argparse
import logging
import datetime as dt

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.gen
from tornado.options import define, options
import tornado.log


define('port', default=2889, help='run on the given port', type=int)


class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.write('Hello world from the HIT service!')


class AssignHandler(tornado.web.RequestHandler):
	
	@tornado.gen.coroutine
	def post(self):
		resp_body = dict()
		try:
			assigns = simplejson.loads(self.request.body)
			if not isinstance(assigns, list):
				raise ValueError

			num_inserted = 0
			for request_body in assigns:
				if 'worker' in request_body.keys() and 'object' in request_body.keys() and 'labels' in request_body.keys() and 'timestamp_utc' in request_body.keys():
					status = yield tornado.gen.Task(self._async_insert, **request_body)
					if status:
						logging.info('[x] [%s] Inserted %s' % (dt.datetime.now().strftime('%I:%M:%p on %B %d, %Y'), request_body))
						num_inserted += 1
				
				resp_body['num_inserted'] = num_inserted	
				resp_body['status_code'] = 200

		except ValueError:
			resp_body['status_code'] = 500
		
		self.write(simplejson.dumps(resp_body))


	def _async_insert(self, worker, object, labels, timestamp_utc, callback):
		""" Insert HIT assignment to a data store asynchronously
		Args:
			worker: ID of the worker who performed the task
			object: ID of the object
			labels: List of assigned labels
			timestamp_utc: UTC Timestamp of the task
		Return:
			boolean indicating the status of the insertion
		"""
		result = False
		if worker == 'foo':
			result = True

		return callback(result)


handlers = [
	(r'/', MainHandler),
	(r'/assign', AssignHandler),
]
settings = dict(
		template_path=os.path.join(os.path.dirname(__file__),'static/templates'),
		debug='True',
	)
application = tornado.web.Application(handlers,**settings)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Launch an HIT server')
	args = parser.parse_args()
	cmd_args = vars(args)

	tornado.log.enable_pretty_logging() # enable pretty console logging
	logging.info('[x] [%s] Starting HIT server on port %s' % (dt.datetime.now().strftime('%I:%M:%p on %B %d, %Y'), options.port))
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()
	