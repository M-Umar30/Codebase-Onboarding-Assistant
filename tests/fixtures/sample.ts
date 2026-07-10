import { Context } from './context'
import type { Handler } from './types'

const DEFAULT_LIMIT = 100

export const makeHandler = (limit: number): Handler => {
  const effective = limit || DEFAULT_LIMIT
  return (ctx: Context) => ctx.json({ effective })
}

function plainFunction(name: string): string {
  return name.trim()
}

export function exportedFunction(): number {
  return DEFAULT_LIMIT
}

export class Service {
  private name: string

  constructor(name: string) {
    this.name = name
  }

  greet(): string {
    return `hello ${this.name}`
  }

  static create(name: string): Service {
    return new Service(name)
  }
}

// Trailing module-level wiring — must be covered by a chunk.
const singleton = Service.create('default')

export default singleton
