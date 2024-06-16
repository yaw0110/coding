import json

from tenant.etlconfig import ETLConfig, ETLDictionary, ETLRegion, ETLInandout, ETLAddress

class TransferSupplier:

    @staticmethod
    def transfer(col_descrs, row_values, value):
        demanderid = row_values[col_descrs.index('id')]
        if value is None or value == '':
            return [{'demanderid': demanderid, 'supplierid': None}]
        try:
            jObj = json.loads(value)
            if len(jObj) <= 0:
                return [{'demanderid': demanderid, 'supplierid': None}]
            arr = []
            for obj in jObj:
                id = obj['id'] if 'id' in obj else ''
                if id == '':
                    id = obj['key'] if 'key' in obj else ''
                if id == '':
                    continue
                arr.append({'demanderid': demanderid, 'supplierid': id})
            return arr
        except Exception as e:
            pass
        return [{'demanderid': demanderid, 'supplierid': None}]


class JiaDun(object):
    tenant_code = '10085068'

    @staticmethod
    def get_etl_config():

        product = [JiaDun.Product, JiaDun.ProductBrand, JiaDun.ProductCategory, JiaDun.ProductDistributionUnit, JiaDun.ProductSingleUnit]
        distributor = [JiaDun.Distributor, JiaDun.DistributorChannelType, JiaDun.DistributorInandout, JiaDun.DistributorOrder, JiaDun.DistributorOrderDetail]
        store = [JiaDun.Store, JiaDun.StoreStoreType, JiaDun.StoreChannelType, JiaDun.StoreInandout]
        other = [JiaDun.OrderSalesType, JiaDun.OrderSalesUnit, JiaDun.Region]


        photo = [JiaDun.StoreInandoutPhoto,JiaDun.StoreInandoutPhoto1]
        # 运行的表
        needRunning = photo + [JiaDun.StoreInandout]

        # return [JiaDun.DistributorOrderDetail]
        return needRunning
    


    class Region(ETLRegion):
        sql = ETLRegion.sql
        config = ETLRegion.config

    class Product(ETLConfig):
        sql = '''
        select productcode, productname, productbrand brandcode, productcategory categorycode, '' specification, distributionunit distributionunitcode, singleunit singleunitcode, unitconverrate unitconverrate
        from kx_kq_product
        '''
        config = {
            'ckTable': 'ods_product'
        }

    class ProductBrand(ETLConfig):
        sql = ETLDictionary.makeSql('1598123939433943040')
        config = ETLDictionary.makeConfig('ods_product_brand')

    class ProductCategory(ETLConfig):
        sql = ETLDictionary.makeSql('1648239495843287040')
        config = ETLDictionary.makeConfig('ods_product_category')

    class ProductDistributionUnit(ETLConfig):
        sql = ETLDictionary.makeSql('896258372821716992')
        config = ETLDictionary.makeConfig('ods_product_distributionunit')

    class ProductSingleUnit(ETLConfig):
        sql = ETLDictionary.makeSql('896258481634545664')
        config = ETLDictionary.makeConfig('ods_product_singleunit')

    class Distributor(ETLConfig):
        sql = '''
            select channelcode distributorcode, channelname distributorname, channeltype channeltypecode, null levelcode,  contactname, contactphone, regionid, address, 1 distributortypecode
            from ka_kq_channelcustomers
            where channelname  not like '%测试%'
                    '''
        config = {
            'ckTable': 'ods_distributor',
            'jsons': ETLAddress.makeJson(),
            'relations': [{
                'col': 'channeltypecode',
                'ckTable': 'ods_distributor_channeltype_relation',
                'transferFunc': lambda col_descrs, row_values, value: [{'channeltypecodetype': 1, 'channeltypecode': value}]
            }, {
                'col': 'regionid',
                'ckTable': 'ods_distributor_region_relation',
                'transferFunc': lambda col_descrs, row_values, value: [{'areaidtype': 1, 'areaid': value, 'customvalue': ''}]
            }]
        }

    class DistributorChannelType(ETLConfig):
        sql = ETLDictionary.makeSql('1598125209460805632')
        config = ETLDictionary.makeConfig('ods_distributor_channeltype')

    class DistributorLevel(ETLConfig):
        pass

    class DistributorInandout(ETLConfig):

        sql = '''
        select inoutid id, platupdatetime, 1 status, channelid distributorid, signintime, cast(duration as Int), signinpictures signinpicture, signinaddress, 2 visittypecode
        from kx_kq_channelinandout
        where signintime >= '2023-01-01 00:00:00'
        '''
        config = {
            'ckTable': 'ods_distributor_inandout',
            'cols': {
                'default': None
            },
            'jsons': ETLInandout.makeJson()
        }

    class DistributorOrder(ETLConfig):
        sql = '''
        select kx_order.id id, kx_order.platupdatetime, 1 status, code ordercode, kx_kq_store.id demanderid, null supplierid,  ordertime, finalamount orderamount, 2 ordertypecode
        from kx_order
        left join kx_kq_store on kx_kq_store.storecode = customercode
        where kx_order.status = 30
        '''
        config = {
            'ckTable': 'ods_distributor_order',
            'cols': {
                'default': None
            },
        }

    class DistributorOrderDetail(ETLConfig):
        sql = '''
        select kx_order_detail.id id, kx_order.platupdatetime, 1 status, orderid, productid, salestype salestypecode, batchunit salesunitcode, batchcount quantity, price unitprice, kx_order_detail.finalamount totalprice
        from kx_order, kx_order_detail
        where kx_order.status = 30 and kx_order.id = kx_order_detail.orderid
        '''
        config = {
            'ckTable': 'ods_distributor_order_detail',
            'cols': {
                'default': None
            },
        }

    class Store(ETLConfig):
        sql = '''
        select storecode, storename, channeltype channeltypecode, storetype storetypecode, contactname, contactphone,regionid, address, supplier
        from kx_kq_store
        where storename not like '%测试%' and address not like '%海洲路%'
        '''
        config = {
            'ckTable': 'ods_store',
            'jsons': ETLAddress.makeJson(),
            'relations': [{
                'col': 'regionid',
                'ckTable': 'ods_store_region_relation',
                'transferFunc': lambda col_descrs, row_values, value: [{'areaidtype': 1, 'areaid': value, 'customvalue': ''}]
            }, {
                'col': 'supplier',
                'ckTable': 'ods_store_supplier',
                'transferFunc': TransferSupplier.transfer
            }]
        }

    class StoreChannelType(ETLConfig):
        sql = ETLDictionary.makeSql('895938457992564736')
        config = ETLDictionary.makeConfig('ods_store_channeltype')

    class StoreStoreType(ETLConfig):
        sql = ETLDictionary.makeSql('1654010977970163712')
        config = ETLDictionary.makeConfig('ods_store_storetype')

    class StoreInandout(ETLConfig):

        # select id, platupdatetime, 1 status, storeid, signintime, cast(duration as Int), signinpictures signinpicture, signinaddress, 1 visittypecode
        # from kx_kq_storeinandout
        # where signintime >= '2023-01-01 00:00:00'

        sql = '''
        select t.id,t.platupdatetime,t.status,t.storeid,store.storename,t.signintime,t.duration,t.signinpicture,t.signinaddress,t.visittypecode
        from (
            select id, platupdatetime, 1 status, storeid, signintime, cast(duration as Int), signinpictures signinpicture, signinaddress, 1 visittypecode
            from kx_kq_storeinandout kks
            where signintime >= '2023-01-01 00:00:00'
            ) t
        left join kx_kq_store store on store.id = t.storeid
        '''
        config = {
            'ckTable': 'ods_store_inandout',
            'cols': {
                'default': None
            },
            'jsons': ETLInandout.makeJson()
        }

    class OrderSalesType(ETLConfig):
        sql = ETLDictionary.makeSql('954254124545871872')
        config = ETLDictionary.makeConfig('ods_order_salestype')

    class OrderSalesUnit(ETLConfig):
        sql = ETLDictionary.makeSql('896258372821716992')
        config = ETLDictionary.makeConfig('ods_order_salesunit')

    class StoreInandoutPhoto(ETLConfig):
        sql = '''
        select t.id, t.platcreatetime, t.status, t.storeid , kks.storename, t.photo, t.phototype,t.submittime
        from (
            select id, platcreatetime, 1 status, storeid, submittime, highcl photo, 2 phototype
            from kx_kq_biscuitca
            where submittime >= '2023-01-01 00:00:00'
            union all
            select id, platcreatetime, 1 status, storeid, submittime, newcl photo, 2 phototype
            from kx_kq_biscuitca
            where submittime >= '2023-01-01 00:00:00'
            union all
            select id, platcreatetime, 1 status, storeid, submittime, boxcl photo, 2 phototype
            from kx_kq_biscuitca
            where submittime >= '2023-01-01 00:00:00'
            union all
            select id, platcreatetime, 1 status, storeid, submittime, mpcl photo, 2 phototype
            from kx_kq_biscuitca
            where submittime >= '2023-01-01 00:00:00'
            union all
            select id, platcreatetime, 1 status, storeid, submittime, pricecl photo, 2 phototype
            from kx_kq_biscuitca
            where submittime >= '2023-01-01 00:00:00'
            ) t
        left join kx_kq_store kks on kks.id = t.storeid
        '''
        config = {
            'name': '拜访采集－终端客户-饼干生动化、面包生动化',
            'ckTable': 'ods_store_inandout_photo',
            'cols': {
                'default': None
            },
            'jsons': ETLInandout.makeJson()
        }

    class StoreInandoutPhoto1(ETLConfig):
        sql = '''
        select t.id,t.platcreatetime,t.status,t.storeid,kks.storename,t.photo,t.phototype,t.submittime
        from (
            select id,platcreatetime,1 status, store_id storeid,createtime submittime, display_photo photo, 2 phototype
            from kx_store_competition_record
            where createtime >= '2023-01-01 00:00:00'
            ) t
        left join kx_kq_store kks on kks.id = t.storeid
        '''
        config = {
            'name': '拜访采集－终端客户-竞品展示',
            'ckTable': 'ods_store_inandout_photo',
            'clear':None,
            'cols': {
                'default': None
            },
            'jsons': ETLInandout.makeJson()
        }