import asyncio
from prisma import Prisma

client = Prisma()

async def main() -> None:
    await client.connect()

    # Create users
    await client.user.create({"id": 1})
    await client.user.create({"id": 2})

    # Create resources
    await client.resource.create({"id": 1, "name": "Resource 1", "type": "video"})
    await client.resource.create({"id": 2, "name": "Resource 2", "type": "video"})
    await client.resource.create({"id": 3, "name": "Resource 3", "type": "test"})
    await client.resource.create({"id": 4, "name": "Resource 4", "type": "video"})
    await client.resource.create({"id": 5, "name": "Resource 5", "type": "video"})

    # Create interactions
    await client.resourceinteraction.create({"id": 1, "userId": 1, "resourceId": 1, "state": 1})
    await client.resourceinteraction.create({"id": 2, "userId": 1, "resourceId": 2, "state": 1})
    await client.resourceinteraction.create({"id": 3, "userId": 2, "resourceId": 1, "state": 1})
    await client.resourceinteraction.create({"id": 4, "userId": 2, "resourceId": 2, "state": 1})

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())