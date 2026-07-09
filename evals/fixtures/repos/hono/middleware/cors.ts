/**
 * @module
 * CORS Middleware for Hono.
 */

import type { Context } from '../../context'
import type { MiddlewareHandler } from '../../types'

type CORSOptions = {
  origin?:
    | string
    | string[]
    | ((
        origin: string,
        c: Context
      ) => Promise<string | undefined | null> | string | undefined | null)
  allowMethods?: string[] | ((origin: string, c: Context) => Promise<string[]> | string[])
  allowHeaders?: string[]
  maxAge?: number
  credentials?: boolean
  exposeHeaders?: string[]
}

export const cors = (options?: CORSOptions): MiddlewareHandler => {
  const opts = {
    origin: '*',
    allowMethods: ['GET', 'HEAD', 'PUT', 'POST', 'DELETE', 'PATCH'],
    allowHeaders: [],
    exposeHeaders: [],
    ...options,
  } satisfies CORSOptions

  const findAllowOrigin = ((optsOrigin) => {
    if (typeof optsOrigin === 'string') {
      if (optsOrigin === '*') {
        return () => optsOrigin
      } else {
        return (origin: string) => (optsOrigin === origin ? origin : null)
      }
    } else if (typeof optsOrigin === 'function') {
      return optsOrigin
    } else {
      return (origin: string) => (optsOrigin.includes(origin) ? origin : null)
    }
  })(opts.origin)

  return async function cors(c, next) {
    function set(key: string, value: string) {
      c.res.headers.set(key, value)
    }

    const allowOrigin = await findAllowOrigin(c.req.header('origin') || '', c)
    if (allowOrigin) {
      set('Access-Control-Allow-Origin', allowOrigin)
    }

    if (opts.credentials) {
      set('Access-Control-Allow-Credentials', 'true')
    }

    if (opts.exposeHeaders?.length) {
      set('Access-Control-Expose-Headers', opts.exposeHeaders.join(','))
    }

    await next()

    if (opts.origin !== '*') {
      c.header('Vary', 'Origin', { append: true })
    }
  }
}
